"""대화형 RAG REPL — 프롬프트에 질문을 입력하면 검색+답변.

두 모드 (--mode):
  - hotpot   : 질문만 입력. HotpotQA validation 풀에서 가장 비슷한 질문을
               찾아 그 샘플의 후보 10단락을 retrieval pool로 사용.
               (실제 corpus 없이 빠르게 RAG 체감용)
  - passages : 질문 + 단락들을 직접 입력. 자기 텍스트로 테스트하고 싶을 때.

학습된 RL 정책이 있으면 그걸로 selection도 가능:
  - top_k_sim : sentence-transformers 유사도 top-k (Naive RAG, default)
  - rl        : 학습된 REINFORCE 정책 + greedy. --ckpt 필요.

사용 예:
    python -m src.ask                                  # hotpot 모드, top_k_sim
    python -m src.ask --k 5                            # top-5
    python -m src.ask --mode passages                  # 단락 직접 입력
    python -m src.ask --policy rl --ckpt models/step_seed42_final.pt
    python -m src.ask --pool-size 1000                 # 더 넓은 retrieval pool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset

from .evaluate import compute_exact_match
from .rewards import compute_answer_f1
from .state_encoder import StateEncoder


PROMPT_QUESTION = "\n질문> "
PROMPT_PASSAGE = "  단락> "
EXIT_TOKENS = {"q", "quit", "exit", ":q"}


def cosine_topk(query: np.ndarray, mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """query (D,) vs mat (N,D) cosine 유사도 top-k 인덱스 + sims 반환."""
    q = query / (np.linalg.norm(query) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    idx = np.argsort(-sims)[:k]
    return idx, sims


def prompt_multiline_passages() -> List[str]:
    """단락을 한 줄에 하나씩 받는다. 빈 줄로 종료."""
    print("  단락을 한 줄에 하나씩 입력. 빈 줄로 종료. (단락이 0개면 이 질문은 건너뜀)")
    passages: List[str] = []
    while True:
        try:
            line = input(PROMPT_PASSAGE)
        except EOFError:
            break
        if not line.strip():
            break
        passages.append(line.strip())
    return passages


def select_passages_topk(
    encoder: StateEncoder,
    question: str,
    passages: List[str],
    k: int,
    verbose: bool = True,
) -> Tuple[List[str], List[str], List[int]]:
    """top-k MiniLM 유사도로 keep할 단락 선택. (kept_texts, kept_titles_or_preview, kept_idx)."""
    if len(passages) <= k:
        return passages, [f"p{i}" for i in range(len(passages))], list(range(len(passages)))
    q_emb = encoder.encode([question])[0]
    p_embs = encoder.encode(passages)
    top, sims = cosine_topk(q_emb, p_embs, k)
    if verbose:
        for rank, i in enumerate(top):
            preview = passages[i][:80].replace("\n", " ")
            print(f"    {rank+1}. sim={sims[i]:.3f} | {preview}...")
    kept = [passages[i] for i in top]
    return kept, [f"p{i}" for i in top], list(top)


def select_passages_rl(
    ckpt_path: str,
    encoder: StateEncoder,
    question: str,
    passages: List[str],
    titles: Optional[List[str]],
    device: str,
    verbose: bool = True,
):
    """학습된 RL 정책으로 selection. titles 없으면 'p0','p1',... 사용."""
    from .agent import REINFORCEAgent
    from .env import RAGEnv
    from .rl_types import Action, ActionKind
    from .state_encoder import encode_sample, expected_state_dim, state_to_vector

    state_dim = expected_state_dim(n_candidates=10, emb_dim=encoder.emb_dim)
    n_actions = 21
    agent = REINFORCEAgent(state_dim=state_dim, n_actions=n_actions, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(ckpt["policy"])
    agent.policy.eval()

    if titles is None:
        titles = [f"p{i}" for i in range(len(passages))]
    # RAGEnv 입력 형태로 sample 만들기 (supporting_facts는 더미)
    sample = {
        "id": "interactive",
        "question": question,
        "answer": "",
        "type": "?",
        "level": "?",
        "supporting_facts": {"title": [], "sent_id": []},
        "context": {"title": titles[:10], "sentences": [[p] for p in passages[:10]]},
    }
    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    state = env.reset(sample)
    encoded = encode_sample(encoder, env.passages, env.question, n_candidates_max=10)
    kept_idx: List[int] = []
    done = False
    while not done:
        sv = state_to_vector(state, encoded, max_steps=env.max_steps)
        st = torch.from_numpy(sv).float()
        mk = torch.tensor(state.valid_actions_mask, dtype=torch.bool)
        a = agent.greedy_action(st, mk)
        action = Action.from_index(a, 10)
        if verbose:
            tag = action.kind.value
            if action.passage_idx is not None:
                t = titles[action.passage_idx]
                preview = passages[action.passage_idx][:60].replace("\n", " ")
                print(f"    step {state.step+1}: {tag:>4} idx={action.passage_idx} ({t}) | {preview}...")
            else:
                print(f"    step {state.step+1}: {tag}")
        # info를 위해 env.step 호출하지만 LLM은 안 부름
        state, _, done, _ = env.step(a)
    kept_idx = list(env._kept_indices)
    kept_texts = [passages[i] for i in kept_idx]
    kept_titles = [titles[i] for i in kept_idx]
    return kept_texts, kept_titles, kept_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hotpot", "passages"], default="hotpot",
                        help="hotpot: Q만 입력 후 유사한 샘플의 단락 사용 / passages: Q+단락 모두 입력")
    parser.add_argument("--policy", choices=["top_k_sim", "rl"], default="top_k_sim",
                        help="selection 정책: 유사도 top-k 또는 학습된 RL")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="--policy rl일 때 정책 체크포인트 경로")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--pool-size", type=int, default=500,
                        help="hotpot 모드에서 매칭 대상으로 둘 validation 샘플 수")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    if args.policy == "rl" and not args.ckpt:
        parser.error("--policy rl 은 --ckpt 가 필요합니다")

    print("[encoder] loading sentence-transformers...")
    encoder = StateEncoder(device=args.device)

    print("[llm] loading Qwen2.5-0.5B...")
    from .llm import QwenAnswerer
    answerer = QwenAnswerer(device=args.device, max_new_tokens=args.max_new_tokens)

    # hotpot 모드면 question pool 인코딩
    pool = None
    pool_q_embs = None
    if args.mode == "hotpot":
        print(f"[pool] HotpotQA validation 처음 {args.pool_size}개 인코딩 중...")
        ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
        pool = ds["validation"].select(range(min(args.pool_size, len(ds["validation"]))))
        pool_questions = [pool[i]["question"] for i in range(len(pool))]
        pool_q_embs = encoder.encode(pool_questions)
        print(f"[pool] {len(pool)} 샘플 준비 완료\n")

    print("=" * 70)
    print(f"  RAG REPL (mode={args.mode}, policy={args.policy}, k={args.k})")
    print("  종료: 빈 입력, q, quit, exit, Ctrl+C")
    print("=" * 70)

    while True:
        try:
            question = input(PROMPT_QUESTION).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break
        if not question or question.lower() in EXIT_TOKENS:
            print("[exit]")
            break

        # 단락 확보
        if args.mode == "passages":
            passages = prompt_multiline_passages()
            titles: Optional[List[str]] = None
            gold_answer: Optional[str] = None
            gold_titles: List[str] = []
        else:  # hotpot
            q_emb = encoder.encode([question])[0]
            top_idx, sims = cosine_topk(q_emb, pool_q_embs, 1)
            best = int(top_idx[0])
            matched = pool[best]
            print(f"  [hotpot] 매칭 sample idx={best}, sim={sims[best]:.3f}")
            print(f"  [hotpot] 매칭된 질문: {matched['question']!r}")
            titles = list(matched["context"]["title"])
            passages = ["".join(s) for s in matched["context"]["sentences"]]
            gold_titles = list(matched["supporting_facts"]["title"])
            gold_answer = matched["answer"]
            print(f"  [hotpot] 후보 단락 {len(titles)}개 (정답: {gold_titles})")

        if not passages:
            print("  (단락 없음, 다음 질문으로)")
            continue

        # selection
        print(f"\n  --- selection ({args.policy}) ---")
        if args.policy == "top_k_sim":
            kept_texts, kept_tags, kept_idx = select_passages_topk(
                encoder, question, passages, args.k, verbose=True
            )
        else:  # rl
            kept_texts, kept_tags, kept_idx = select_passages_rl(
                args.ckpt, encoder, question, passages, titles, args.device, verbose=True
            )

        # 답변 생성
        print(f"\n  [Qwen2.5-0.5B 생성 중... (kept={len(kept_texts)})]")
        pred = answerer(question, kept_texts)
        print(f"\n  답변 : {pred!r}")

        # 메트릭 (gold가 있을 때만)
        if gold_answer:
            f1 = compute_answer_f1(pred, gold_answer)
            em = compute_exact_match(pred, gold_answer)
            kept_titles_set = set(titles[i] for i in kept_idx) if titles else set()
            sup_set = kept_titles_set & set(gold_titles)
            sup_p = len(sup_set) / max(1, len(kept_titles_set))
            sup_r = len(sup_set) / max(1, len(gold_titles))
            sup_f1 = 2 * sup_p * sup_r / max(1e-9, sup_p + sup_r) if (sup_p + sup_r) else 0
            print(f"  정답 : {gold_answer!r}")
            print(f"  F1   : {f1:.3f}   EM: {em:.0f}   sup_F1: {sup_f1:.3f}")


if __name__ == "__main__":
    main()
