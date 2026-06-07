"""최종 결과 표(table1.csv) + 학습곡선(learning_curves.png) 생성.

- baseline_*.json (Naive RAG 변형) + eval_rl_*.json (학습된 정책) 취합 → table1.csv
- logs/train_{step,sparse}_seed*.log 에서 dev_F1 추세 파싱 → learning_curves.png

실행:
    python -m src.build_results
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def collect_table() -> list[dict]:
    rows = []
    # baseline_*.json
    for fp in sorted(glob.glob(str(RESULTS / "baseline_*.json"))):
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        rows.append({
            "method": d.get("variant", Path(fp).stem),
            "answer_f1": d.get("answer_f1"),
            "answer_f1_std": d.get("answer_f1_std"),
            "exact_match": d.get("exact_match"),
            "support_f1": d.get("support_f1"),
            "avg_n_kept": d.get("avg_n_kept"),
            "n": d.get("n"),
        })
    # eval_rl_*.json
    for fp in sorted(glob.glob(str(RESULTS / "eval_rl_*.json"))):
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        rows.append({
            "method": d.get("variant", Path(fp).stem),
            "answer_f1": d.get("answer_f1"),
            "answer_f1_std": d.get("answer_f1_std"),
            "exact_match": d.get("exact_match"),
            "support_f1": d.get("support_f1"),
            "avg_n_kept": d.get("avg_n_kept"),
            "n": d.get("n"),
        })
    return rows


def parse_dev_curve(log_path: Path) -> tuple[list[int], list[float], float]:
    """로그에서 'BC 직후 dev_F1' + '[dev epN] dev_F1=...' 파싱."""
    if not log_path.exists():
        return [], [], None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    bc = re.search(r"BC 직후 dev_F1 = ([\d.]+)", text)
    bc_f1 = float(bc.group(1)) if bc else None
    eps, f1s = [], []
    if bc_f1 is not None:
        eps.append(0)
        f1s.append(bc_f1)
    for m in re.finditer(r"\[dev ep(\d+)\] dev_F1=([\d.]+)", text):
        eps.append(int(m.group(1)))
        f1s.append(float(m.group(2)))
    return eps, f1s, bc_f1


def main() -> None:
    # ---- table1.csv ----
    rows = collect_table()
    # 보기 좋은 순서
    order = {"oracle_k3": 0, "oracle": 0, "top_k_sim_k3": 1, "use_all": 2,
             "random_k3": 3}
    def keyf(r):
        m = r["method"]
        if m.startswith("rl_step"): return 10
        if m.startswith("rl_sparse"): return 11
        if m.startswith("rl_"): return 12
        return order.get(m, 5)
    rows.sort(key=keyf)

    csv_path = RESULTS / "table1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "answer_f1", "answer_f1_std", "exact_match",
                    "support_f1", "avg_n_kept", "n"])
        for r in rows:
            w.writerow([r["method"],
                        round(r["answer_f1"], 4) if r["answer_f1"] is not None else "",
                        round(r["answer_f1_std"], 4) if r["answer_f1_std"] is not None else "",
                        round(r["exact_match"], 4) if r["exact_match"] is not None else "",
                        round(r["support_f1"], 4) if r["support_f1"] is not None else "",
                        round(r["avg_n_kept"], 2) if r["avg_n_kept"] is not None else "",
                        r["n"]])
    print(f"[saved] {csv_path}")
    print(f"\n{'method':<24}{'ans_F1':>9}{'EM':>7}{'sup_F1':>8}{'kept':>6}{'n':>6}")
    print("-" * 62)
    for r in rows:
        print(f"{r['method']:<24}{r['answer_f1']:>9.3f}{r['exact_match']:>7.3f}"
              f"{r['support_f1']:>8.3f}{r['avg_n_kept']:>6.1f}{r['n']:>6}")

    # ---- learning_curves.png (dev F1: step vs sparse) ----
    step_eps, step_f1, _ = parse_dev_curve(LOGS / "train_step_seed42.log")
    sparse_eps, sparse_f1, _ = parse_dev_curve(LOGS / "train_sparse_seed42.log")

    if step_eps or sparse_eps:
        fig, ax = plt.subplots(figsize=(8, 5))
        if step_eps:
            ax.plot(step_eps, step_f1, "-o", label="Step-wise RL", color="C0")
        if sparse_eps:
            ax.plot(sparse_eps, sparse_f1, "-s", label="Sparse RL", color="C1")
        # baseline 기준선
        ax.axhline(0.370, ls="--", c="green", alpha=0.6, label="Naive RAG (cosine top-3)")
        ax.axhline(0.557, ls=":", c="gray", alpha=0.6, label="Oracle (upper bound)")
        ax.set_xlabel("training episode")
        ax.set_ylabel("dev answer_F1 (greedy, n=50)")
        ax.set_title("Step-wise vs Sparse reward — dev F1 (seed 42)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        out = RESULTS / "learning_curves.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
