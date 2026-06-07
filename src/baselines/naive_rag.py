"""Naive RAG baseline — 학습 없음.

4가지 변형:
  1. use_all   : 후보 단락 10개 전부 LLM context로 전달 (recall=1, noise 최대)
  2. top_k_sim : sentence-transformers로 question vs passage 유사도 → top-k
  3. random    : 무작위 k개 keep (lower bound 디버깅)
  4. oracle    : supporting_facts.title로 정답 단락만 keep (학습 upper bound)

사용 예 (CLI):
    python -m src.baselines.naive_rag --variant use_all --n 200
    python -m src.baselines.naive_rag --variant top_k_sim --k 3 --n 200
    python -m src.baselines.naive_rag --variant oracle --n 200
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from datasets import load_dataset

from ..evaluate import evaluate


ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- answer_fn 빌더 ----------


def make_use_all_fn(answerer):
    """후보 10개 전부 LLM에 전달."""
    def _fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        titles = sample["context"]["title"]
        sents = sample["context"]["sentences"]
        passages = ["".join(s) for s in sents]
        pred = answerer(sample["question"], passages)
        return {"predicted_answer": pred, "kept_titles": list(titles), "n_steps": len(titles)}
    return _fn


def make_top_k_sim_fn(answerer, encoder, k: int = 3):
    """sentence-transformers 유사도 top-k."""
    def _fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        titles = sample["context"]["title"]
        sents = sample["context"]["sentences"]
        passages = ["".join(s) for s in sents]
        # 인코딩
        q_emb = encoder.encode([sample["question"]])  # (1, D)
        p_emb = encoder.encode(passages)              # (N, D)
        # cosine 유사도 (encoder 출력은 정규화 안 되었을 수 있음 → 직접 정규화)
        q = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)
        p = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-9)
        sims = (p @ q.T).squeeze(1)  # (N,)
        top_idx = np.argsort(-sims)[:k]
        kept_titles = [titles[i] for i in top_idx]
        kept_texts = [passages[i] for i in top_idx]
        pred = answerer(sample["question"], kept_texts)
        return {"predicted_answer": pred, "kept_titles": kept_titles, "n_steps": k}
    return _fn


def make_random_keep_fn(answerer, k: int = 3, seed: int = 0):
    """무작위 k개 keep."""
    rng = random.Random(seed)
    def _fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        titles = sample["context"]["title"]
        sents = sample["context"]["sentences"]
        passages = ["".join(s) for s in sents]
        idx = rng.sample(range(len(titles)), min(k, len(titles)))
        kept_titles = [titles[i] for i in idx]
        kept_texts = [passages[i] for i in idx]
        pred = answerer(sample["question"], kept_texts)
        return {"predicted_answer": pred, "kept_titles": kept_titles, "n_steps": len(idx)}
    return _fn


def make_oracle_fn(answerer):
    """upper bound: supporting_facts에 있는 정답 단락만 LLM에 전달."""
    def _fn(sample: Dict[str, Any]) -> Dict[str, Any]:
        titles = sample["context"]["title"]
        sents = sample["context"]["sentences"]
        gold_set = set(sample["supporting_facts"]["title"])
        kept_titles, kept_texts = [], []
        for t, ss in zip(titles, sents):
            if t in gold_set:
                kept_titles.append(t)
                kept_texts.append("".join(ss))
        pred = answerer(sample["question"], kept_texts)
        return {"predicted_answer": pred, "kept_titles": kept_titles, "n_steps": len(kept_titles)}
    return _fn


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["use_all", "top_k_sim", "random", "oracle"], required=True)
    parser.add_argument("--k", type=int, default=3, help="top_k_sim/random에서 선택할 단락 수")
    parser.add_argument("--n", type=int, default=200, help="HotpotQA validation 평가 샘플 수")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"[setup] variant={args.variant} k={args.k} n={args.n} device={args.device}")

    print("[data] HotpotQA load...")
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    eval_ds = ds[args.split].shuffle(seed=args.seed).select(range(args.n))
    print(f"[data] eval samples = {len(eval_ds)}")

    print("[llm] loading Qwen2.5-0.5B...")
    from ..llm import QwenAnswerer
    answerer = QwenAnswerer(device=args.device, max_new_tokens=args.max_new_tokens)

    if args.variant == "use_all":
        fn = make_use_all_fn(answerer)
    elif args.variant == "top_k_sim":
        from ..state_encoder import StateEncoder
        print("[encoder] loading sentence-transformers...")
        encoder = StateEncoder(device=args.device)
        fn = make_top_k_sim_fn(answerer, encoder, k=args.k)
    elif args.variant == "oracle":
        fn = make_oracle_fn(answerer)
    else:  # random
        fn = make_random_keep_fn(answerer, k=args.k, seed=args.seed)

    desc = args.variant + (f"_k{args.k}" if args.variant != "use_all" else "")
    print(f"\n=== evaluating: {desc} ===")
    result = evaluate(fn, eval_ds, n_samples=args.n, desc=desc)
    result["variant"] = desc
    result["n_eval"] = args.n

    # 저장
    out_path = RESULTS_DIR / f"baseline_{desc}_n{args.n}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] {desc}")
    print(f"  answer_F1   = {result['answer_f1']:.3f} ± {result['answer_f1_std']:.3f}")
    print(f"  exact_match = {result['exact_match']:.3f}")
    print(f"  support_F1  = {result['support_f1']:.3f}")
    print(f"  avg_n_kept  = {result['avg_n_kept']:.2f}")
    print(f"  by_type     = { {k: v['f1'] for k,v in result['by_type'].items()} }")
    print(f"  by_level    = { {k: v['f1'] for k,v in result['by_level'].items()} }")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
