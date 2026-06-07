"""범용 평가기 — answerer 함수 + dataset → 평균 answer_F1 + 부수 지표.

answerer signature:
    answerer(sample) -> {predicted_answer, kept_titles, n_steps}

이렇게 sample 단위로 호출해야 baseline마다 다른 동작 (use-all / top-k /
random policy) 을 표현할 수 있다.

지표:
  - answer_F1 (mean) — 핵심
  - exact_match (mean)
  - supporting_fact_F1 — kept_titles vs supporting_facts.title
  - avg_n_kept — 평균 keep 단락 수
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from datasets import load_dataset

from .rewards import _normalize_answer, compute_answer_f1


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# AnswerFn: sample dict → result dict
AnswerFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def load_local_eval(path: str) -> List[Dict[str, Any]]:
    """로컬 HotpotQA 원본(raw) JSON → HF datasets 형식(dict-of-lists)으로 변환.

    원본 포맷:
      supporting_facts: [[title, sent_id], ...]
      context:          [[title, [sentences]], ...]
      id 키:            _id
    변환 후(우리 env/평가가 기대하는 형식):
      supporting_facts: {"title": [...], "sent_id": [...]}
      context:          {"title": [...], "sentences": [[...], ...]}
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for s in raw:
        sf = s["supporting_facts"]
        ctx = s["context"]
        # 이미 HF 형식(dict)이면 그대로, raw 형식(list)이면 변환
        if isinstance(sf, dict):
            sf_conv = {"title": list(sf["title"]), "sent_id": list(sf["sent_id"])}
        else:
            sf_conv = {"title": [x[0] for x in sf], "sent_id": [x[1] for x in sf]}
        if isinstance(ctx, dict):
            ctx_conv = {"title": list(ctx["title"]), "sentences": list(ctx["sentences"])}
        else:
            ctx_conv = {"title": [c[0] for c in ctx], "sentences": [c[1] for c in ctx]}
        out.append({
            "id": s.get("_id", s.get("id", "")),
            "question": s["question"],
            "answer": s["answer"],
            "type": s.get("type", "?"),
            "level": s.get("level", "?"),
            "supporting_facts": sf_conv,
            "context": ctx_conv,
        })
    return out


def compute_exact_match(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def compute_support_f1(pred_titles: List[str], gold_titles: List[str]) -> float:
    """Supporting fact F1 (title 단위)."""
    if not pred_titles and not gold_titles:
        return 1.0
    if not pred_titles or not gold_titles:
        return 0.0
    pred_set = set(pred_titles)
    gold_set = set(gold_titles)
    common = pred_set & gold_set
    if not common:
        return 0.0
    p = len(common) / len(pred_set)
    r = len(common) / len(gold_set)
    return 2 * p * r / (p + r)


def evaluate(
    answer_fn: AnswerFn,
    dataset,
    n_samples: int = None,
    log_every: int = 50,
    desc: str = "eval",
) -> Dict[str, Any]:
    """answer_fn을 dataset에 돌려서 평균 메트릭 + hop별 breakdown 반환.

    answer_fn은 sample 당 한 번만 호출됨 (LLM 비용 절약).
    """
    n_total = len(dataset) if n_samples is None else min(n_samples, len(dataset))
    f1s, ems, sup_f1s, n_kept_list, n_steps_list = [], [], [], [], []
    per_type: Dict[str, List[float]] = {}
    per_level: Dict[str, List[float]] = {}
    t0 = time.time()

    for i in range(n_total):
        sample = dataset[i]
        out = answer_fn(sample)
        pred = out.get("predicted_answer", "")
        kept_titles = out.get("kept_titles", [])
        n_steps = out.get("n_steps", 0)

        gold_answer = sample["answer"]
        gold_titles = sample["supporting_facts"]["title"]

        f1 = compute_answer_f1(pred, gold_answer)
        f1s.append(f1)
        ems.append(compute_exact_match(pred, gold_answer))
        sup_f1s.append(compute_support_f1(kept_titles, gold_titles))
        n_kept_list.append(len(kept_titles))
        n_steps_list.append(n_steps)

        per_type.setdefault(sample.get("type", "?"), []).append(f1)
        per_level.setdefault(sample.get("level", "?"), []).append(f1)

        if (i + 1) % log_every == 0 or i == 0:
            ips = (i + 1) / max(1e-6, time.time() - t0)
            print(
                f"  [{desc} {i+1:4d}/{n_total}] f1={np.mean(f1s):.3f} "
                f"em={np.mean(ems):.3f} sup_f1={np.mean(sup_f1s):.3f} "
                f"kept={np.mean(n_kept_list):.1f} ips={ips:.1f}"
            )

    return {
        "n": n_total,
        "answer_f1": float(np.mean(f1s)),
        "answer_f1_std": float(np.std(f1s)),
        "exact_match": float(np.mean(ems)),
        "support_f1": float(np.mean(sup_f1s)),
        "avg_n_kept": float(np.mean(n_kept_list)),
        "avg_n_steps": float(np.mean(n_steps_list)),
        "elapsed_sec": time.time() - t0,
        "by_type": {k: {"f1": float(np.mean(v)), "n": len(v)} for k, v in per_type.items()},
        "by_level": {k: {"f1": float(np.mean(v)), "n": len(v)} for k, v in per_level.items()},
    }
