"""3-seed 집계 — 학습곡선(mean±std 음영) + 최종 표(평균±표준편차).

- 곡선: logs/train_{step,sparse}_seed{42,123,7}.log 의 dev_F1 파싱 → seed 평균±std 음영.
- 표: results/eval_rl_{step,sparse}_seed{seed}_best_greedy_n200.json 들을 모아
      step / sparse 각각 seed 평균±std.

실행 (학습 + 각 seed run_eval 후):
    python -m src.aggregate_results
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
SEEDS = [42, 123, 7]


def parse_dev_curve(log_path: Path):
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    curve = {}
    bc = re.search(r"BC 직후 dev_F1 = ([\d.]+)", text)
    if bc:
        curve[0] = float(bc.group(1))
    for m in re.finditer(r"\[dev ep(\d+)\] dev_F1=([\d.]+)", text):
        curve[int(m.group(1))] = float(m.group(2))
    return curve


def aggregate_curves(tag: str):
    """seed별 dev curve → 공통 episode 축에서 mean/std."""
    curves = [parse_dev_curve(LOGS / f"train_{tag}_seed{s}.log") for s in SEEDS]
    curves = [c for c in curves if c]
    if not curves:
        return None
    eps = sorted(set().union(*[set(c.keys()) for c in curves]))
    mean, std, n = [], [], []
    for e in eps:
        vals = [c[e] for c in curves if e in c]
        mean.append(np.mean(vals))
        std.append(np.std(vals))
        n.append(len(vals))
    return np.array(eps), np.array(mean), np.array(std), n


def plot_curves():
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"step": "C0", "sparse": "C1"}
    labels = {"step": "Step-wise RL", "sparse": "Sparse RL"}
    for tag in ["step", "sparse"]:
        agg = aggregate_curves(tag)
        if agg is None:
            continue
        eps, mean, std, _ = agg
        ax.plot(eps, mean, "-o", color=colors[tag], label=f"{labels[tag]} (mean of {len(SEEDS)} seeds)")
        ax.fill_between(eps, mean - std, mean + std, color=colors[tag], alpha=0.2)
    ax.axhline(0.370, ls="--", c="green", alpha=0.6, label="Naive RAG (cosine top-3)")
    ax.axhline(0.557, ls=":", c="gray", alpha=0.6, label="Oracle (upper bound)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("dev answer_F1 (greedy, n=50)")
    ax.set_title(f"Step-wise vs Sparse — dev F1 ({len(SEEDS)} seeds, mean±std)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    out = RESULTS / "learning_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


def aggregate_table():
    """eval json들을 step/sparse별로 모아 seed 평균±std."""
    print(f"\n{'method':<16}{'answer_F1':>16}{'support_F1':>16}{'EM':>10}  (seeds)")
    print("-" * 64)
    summary = {}
    for tag in ["step", "sparse"]:
        f1s, sups, ems, seeds_found = [], [], [], []
        for s in SEEDS:
            cands = glob.glob(str(RESULTS / f"eval_rl_{tag}_seed{s}_best_*_n200.json"))
            if not cands:
                continue
            d = json.loads(Path(cands[0]).read_text(encoding="utf-8"))
            f1s.append(d["answer_f1"]); sups.append(d["support_f1"]); ems.append(d["exact_match"])
            seeds_found.append(s)
        if not f1s:
            print(f"{tag:<16}  (no eval json yet)")
            continue
        summary[tag] = {
            "answer_f1_mean": float(np.mean(f1s)), "answer_f1_std": float(np.std(f1s)),
            "support_f1_mean": float(np.mean(sups)), "support_f1_std": float(np.std(sups)),
            "exact_match_mean": float(np.mean(ems)),
            "seeds": seeds_found,
        }
        print(f"{tag:<16}{np.mean(f1s):>8.3f}±{np.std(f1s):<6.3f}"
              f"{np.mean(sups):>8.3f}±{np.std(sups):<6.3f}{np.mean(ems):>10.3f}  {seeds_found}")
    (RESULTS / "table1_3seed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[saved] {RESULTS / 'table1_3seed.json'}")


def main() -> None:
    plot_curves()
    aggregate_table()


if __name__ == "__main__":
    main()
