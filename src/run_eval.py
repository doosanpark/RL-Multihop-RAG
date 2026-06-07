"""통합 평가 entrypoint — RL 정책 / Naive RAG 변형 모두 동일 API로 비교.

지원 변형 (CLI --variant):
  - oracle        : supporting_facts 정답 단락만 사용 (학습 upper bound)
  - use_all       : 후보 10개 전부 LLM에 전달
  - top_k_sim     : sentence-transformers 유사도 top-k (마스터 컨텍스트의 Naive RAG)
  - random        : 무작위 k개 keep (lower bound)
  - rl            : 학습된 REINFORCE 정책 (--ckpt 필요)

모든 변형은 sample 단위 `answer_fn(sample) -> {predicted_answer, kept_titles, n_steps}`
시그니처를 따르므로 평가 표에서 1:1 비교 가능.

예:
    python -m src.run_eval --variant top_k_sim --k 3 --n 200
    python -m src.run_eval --variant rl --ckpt models/step_seed42_final.pt --n 200
    python -m src.run_eval --variant rl --ckpt models/step_seed42_final.pt --policy sample --n 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from datasets import load_dataset

from .agent import REINFORCEAgent
from .baselines.naive_rag import (
    make_oracle_fn,
    make_random_keep_fn,
    make_top_k_sim_fn,
    make_use_all_fn,
)
from .env import RAGEnv
from .evaluate import evaluate
from .state_encoder import StateEncoder, encode_sample, expected_state_dim, state_to_vector


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_rl_policy_fn(
    ckpt_path: str,
    encoder: StateEncoder,
    answerer,
    device: str = "cpu",
    policy_mode: str = "greedy",
):
    """학습된 REINFORCE 정책으로 매 sample마다 rollout → answer_fn 시그니처."""
    state_dim = expected_state_dim(n_candidates=10, emb_dim=encoder.emb_dim)
    n_actions = 21
    agent = REINFORCEAgent(state_dim=state_dim, n_actions=n_actions, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(ckpt["policy"])
    agent.value.load_state_dict(ckpt["value"])
    agent.policy.eval()
    agent.value.eval()

    env = RAGEnv(n_candidates=10, max_steps=10, answerer=answerer)

    def _fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        state = env.reset(sample)
        env.answerer = answerer  # 매 episode에 동일 LLM 함수
        encoded = encode_sample(encoder, env.passages, env.question, n_candidates_max=10)
        done = False
        info: Dict[str, Any] = {}
        while not done:
            sv = state_to_vector(state, encoded, max_steps=env.max_steps)
            state_t = torch.from_numpy(sv).float()
            mask_t = torch.tensor(state.valid_actions_mask, dtype=torch.bool)
            if policy_mode == "greedy":
                action_idx = agent.greedy_action(state_t, mask_t)
            else:
                action_idx, _ = agent.select_action(state_t, mask_t)
            state, _, done, info = env.step(action_idx)
        return {
            "predicted_answer": info.get("predicted_answer", ""),
            "kept_titles": info.get("kept_titles", []),
            "n_steps": info.get("total_steps", env._step_count),
        }

    return _fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["oracle", "use_all", "top_k_sim", "random", "rl"],
        required=True,
    )
    parser.add_argument("--k", type=int, default=3, help="top_k_sim / random에서 keep 단락 수")
    parser.add_argument("--ckpt", type=str, default=None, help="rl variant의 정책 체크포인트 경로")
    parser.add_argument("--policy", choices=["greedy", "sample"], default="greedy",
                        help="rl variant: argmax(greedy) vs 확률 샘플링")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--eval-file", type=str, default=None,
                        help="로컬 HotpotQA-포맷 JSON 평가셋 (예: data/eval/sports.json). "
                             "지정 시 HF hotpot_qa 대신 이 파일 사용 (cross-domain transfer).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.variant == "rl" and not args.ckpt:
        parser.error("--variant rl 은 --ckpt 가 필요합니다")

    print(f"[setup] variant={args.variant} n={args.n} device={args.device}")
    if args.eval_file:
        import random as _random
        from .evaluate import load_local_eval
        print(f"[data] 로컬 평가셋 로드: {args.eval_file}")
        data = load_local_eval(args.eval_file)
        _random.Random(args.seed).shuffle(data)
        eval_ds = data[: args.n]
        domain = Path(args.eval_file).stem
        print(f"[data] {len(data)}개 중 {len(eval_ds)}개 평가 (domain={domain})")
    else:
        print("[data] HotpotQA load...")
        ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
        eval_ds = ds[args.split].shuffle(seed=args.seed).select(range(args.n))
        domain = "hotpotqa"

    print("[llm] loading Qwen2.5-0.5B...")
    from .llm import QwenAnswerer
    answerer = QwenAnswerer(device=args.device, max_new_tokens=args.max_new_tokens)

    if args.variant == "oracle":
        fn = make_oracle_fn(answerer)
        desc = "oracle"
    elif args.variant == "use_all":
        fn = make_use_all_fn(answerer)
        desc = "use_all"
    elif args.variant == "top_k_sim":
        print("[encoder] loading sentence-transformers (for top-k sim)...")
        encoder = StateEncoder(device=args.device)
        fn = make_top_k_sim_fn(answerer, encoder, k=args.k)
        desc = f"top_k_sim_k{args.k}"
    elif args.variant == "random":
        fn = make_random_keep_fn(answerer, k=args.k, seed=args.seed)
        desc = f"random_k{args.k}"
    else:  # rl
        print(f"[encoder] loading sentence-transformers (for RL state)...")
        encoder = StateEncoder(device=args.device)
        print(f"[policy] loading checkpoint {args.ckpt}...")
        fn = make_rl_policy_fn(
            args.ckpt, encoder, answerer, device=args.device, policy_mode=args.policy
        )
        ckpt_name = Path(args.ckpt).stem
        desc = f"rl_{ckpt_name}_{args.policy}"

    print(f"\n=== evaluating: {desc} on {domain} (n={len(eval_ds)}) ===")
    result = evaluate(fn, eval_ds, n_samples=args.n, desc=desc)
    result["variant"] = desc
    result["domain"] = domain
    result["n_eval"] = len(eval_ds)
    if args.variant == "rl":
        result["ckpt"] = args.ckpt
        result["policy_mode"] = args.policy

    # hotpotqa가 아니면 도메인을 파일명에 포함 (충돌 방지)
    prefix = f"eval_{desc}" if domain == "hotpotqa" else f"eval_{domain}_{desc}"
    out_path = RESULTS_DIR / f"{prefix}_n{len(eval_ds)}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] {desc}")
    print(f"  answer_F1   = {result['answer_f1']:.3f} ± {result['answer_f1_std']:.3f}")
    print(f"  exact_match = {result['exact_match']:.3f}")
    print(f"  support_F1  = {result['support_f1']:.3f}")
    print(f"  avg_n_kept  = {result['avg_n_kept']:.2f}")
    print(f"  avg_n_steps = {result['avg_n_steps']:.2f}")
    print(f"  by_type     = { {k: round(v['f1'],3) for k,v in result['by_type'].items()} }")
    print(f"  by_level    = { {k: round(v['f1'],3) for k,v in result['by_level'].items()} }")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
