"""기본 RAG 한 샘플 데모.

사용 예:
    python -m src.demo                              # 랜덤 1 샘플
    python -m src.demo --index 7                    # validation[7]
    python -m src.demo --variant use_all            # 10개 전부 LLM에 넘김
    python -m src.demo --variant top_k_sim --k 3    # MiniLM 유사도 top-3
    python -m src.demo --variant oracle             # 정답 단락만 (upper bound)
    python -m src.demo --question "..." --passages "..." "..."  # 직접 입력

기본 RAG (top_k_sim)에서 다음을 한눈에 보여줌:
  - 질문 / 정답
  - sentence-transformers가 뽑은 top-k 단락 (유사도 점수 포함)
  - 정답 단락 (supporting_facts)
  - Qwen이 생성한 답변
  - answer_F1, support_F1
"""

from __future__ import annotations

import argparse
import random
from typing import List

import numpy as np
import torch
from datasets import load_dataset

from .evaluate import compute_exact_match
from .rewards import compute_answer_f1


def cosine_topk(q_emb: np.ndarray, p_embs: np.ndarray, k: int) -> List[int]:
    q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    p = p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-9)
    sims = (p @ q)
    return list(np.argsort(-sims)[:k]), sims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None,
                        help="validation split의 인덱스. 미지정 시 랜덤.")
    parser.add_argument("--variant", choices=["use_all", "top_k_sim", "oracle"],
                        default="top_k_sim")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--question", type=str, default=None,
                        help="질문 직접 입력 (지정 시 --passages도 필요)")
    parser.add_argument("--passages", type=str, nargs="*", default=None,
                        help="단락들 직접 입력 (지정 시 --question도 필요)")
    args = parser.parse_args()

    print("[llm] loading Qwen2.5-0.5B...")
    from .llm import QwenAnswerer
    answerer = QwenAnswerer(device=args.device, max_new_tokens=32)

    # ---------- 직접 입력 모드 ----------
    if args.question and args.passages:
        print("\n[모드] 사용자 직접 입력")
        if args.variant == "top_k_sim":
            from .state_encoder import StateEncoder
            print("[encoder] loading sentence-transformers...")
            encoder = StateEncoder(device=args.device)
            q_emb = encoder.encode([args.question])[0]
            p_embs = encoder.encode(args.passages)
            top_idx, sims = cosine_topk(q_emb, p_embs, args.k)
            kept = [args.passages[i] for i in top_idx]
            print(f"\nQ: {args.question}")
            print(f"\n[top-{args.k} retrieved]")
            for rank, i in enumerate(top_idx):
                preview = args.passages[i][:80].replace("\n", " ")
                print(f"  {rank+1}. (sim={sims[i]:.3f}) {preview}...")
        else:
            kept = list(args.passages)
            print(f"\nQ: {args.question}")
            print(f"[passages={len(kept)}개 LLM에 전달]")
        pred = answerer(args.question, kept)
        print(f"\n[Qwen2.5-0.5B 답변] {pred!r}")
        return

    # ---------- HotpotQA 샘플 모드 ----------
    print("[data] HotpotQA validation load...")
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    val = ds["validation"]
    if args.index is None:
        rng = random.Random(args.seed)
        idx = rng.randrange(len(val))
    else:
        idx = args.index
    sample = val[idx]

    titles = sample["context"]["title"]
    passages = ["".join(s) for s in sample["context"]["sentences"]]
    gold_titles = sample["supporting_facts"]["title"]
    gold_answer = sample["answer"]

    print(f"\n{'='*70}")
    print(f"[샘플 #{idx}] type={sample['type']}, level={sample['level']}")
    print(f"{'='*70}")
    print(f"Q: {sample['question']}")
    print(f"\n정답: {gold_answer!r}")
    print(f"정답 단락 (supporting_facts): {gold_titles}")
    print(f"\n후보 단락 ({len(titles)}개):")
    for i, t in enumerate(titles):
        gold = " ★" if t in gold_titles else "  "
        print(f"  [{i:>2}]{gold} {t}")

    # 변형별 selection
    print(f"\n--- variant: {args.variant} ---")
    if args.variant == "use_all":
        kept_idx = list(range(len(titles)))
    elif args.variant == "oracle":
        kept_idx = [i for i, t in enumerate(titles) if t in set(gold_titles)]
    else:  # top_k_sim
        from .state_encoder import StateEncoder
        print("[encoder] loading sentence-transformers...")
        encoder = StateEncoder(device=args.device)
        q_emb = encoder.encode([sample["question"]])[0]
        p_embs = encoder.encode(passages)
        kept_idx, sims = cosine_topk(q_emb, p_embs, args.k)
        print(f"\nMiniLM 유사도 ranking (top-{args.k} keep):")
        ranked = sorted(range(len(titles)), key=lambda i: -sims[i])
        for rank, i in enumerate(ranked[: args.k + 2]):  # top-(k+2)까지 표시
            gold = " ★" if titles[i] in set(gold_titles) else "  "
            kept_mark = "KEEP" if i in kept_idx else "    "
            print(f"  {rank+1:>2}. sim={sims[i]:.3f}{gold} [{kept_mark}] {titles[i]}")

    kept_titles = [titles[i] for i in kept_idx]
    kept_texts = [passages[i] for i in kept_idx]
    print(f"\nLLM에 전달되는 단락: {kept_titles}")

    # 답변 생성
    print(f"\n[Qwen2.5-0.5B 생성 중...]")
    pred = answerer(sample["question"], kept_texts)

    # 메트릭
    f1 = compute_answer_f1(pred, gold_answer)
    em = compute_exact_match(pred, gold_answer)
    sup_set = set(kept_titles) & set(gold_titles)
    sup_p = len(sup_set) / max(1, len(kept_titles))
    sup_r = len(sup_set) / max(1, len(gold_titles))
    sup_f1 = 2 * sup_p * sup_r / max(1e-9, sup_p + sup_r) if (sup_p + sup_r) else 0

    print(f"\n{'='*70}")
    print(f"답변      : {pred!r}")
    print(f"정답      : {gold_answer!r}")
    print(f"answer_F1 : {f1:.3f}    exact_match: {em:.0f}")
    print(f"support_F1: {sup_f1:.3f}  (정답 단락 중 {len(sup_set)}/{len(gold_titles)}개를 keep)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
