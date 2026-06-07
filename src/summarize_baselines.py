"""results/baseline_*.json을 모아서 비교 표 + csv 출력.

실행:
    python -m src.summarize_baselines
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def main() -> None:
    files = sorted(glob.glob(str(RESULTS_DIR / "baseline_*.json")))
    rows = []
    for fp in files:
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        rows.append(d)

    if not rows:
        print("(no baseline_*.json yet)")
        return

    # 정렬: oracle, top_k_sim, use_all, random 순서 (의미적 순서)
    order = {"oracle": 0, "top_k_sim_k3": 1, "use_all": 2, "random_k3": 3}
    rows.sort(key=lambda d: order.get(d.get("variant", "?"), 99))

    print(f"\n{'variant':<14} {'n':>4} {'F1':>8} {'EM':>6} {'sup_F1':>7} {'kept':>5}  by_type / by_level")
    print("-" * 100)
    for d in rows:
        bt = {k: f"{v['f1']:.2f}" for k, v in d.get("by_type", {}).items()}
        bl = {k: f"{v['f1']:.2f}" for k, v in d.get("by_level", {}).items()}
        print(
            f"{d['variant']:<14} {d['n']:>4} "
            f"{d['answer_f1']:>8.3f} {d['exact_match']:>6.3f} "
            f"{d['support_f1']:>7.3f} {d['avg_n_kept']:>5.1f}  "
            f"type={bt}  level={bl}"
        )

    # csv
    csv_path = RESULTS_DIR / "baseline_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "answer_f1", "answer_f1_std",
                    "exact_match", "support_f1", "avg_n_kept", "avg_n_steps", "elapsed_sec"])
        for d in rows:
            w.writerow([
                d.get("variant"), d.get("n"),
                round(d.get("answer_f1", 0), 4),
                round(d.get("answer_f1_std", 0), 4),
                round(d.get("exact_match", 0), 4),
                round(d.get("support_f1", 0), 4),
                round(d.get("avg_n_kept", 0), 2),
                round(d.get("avg_n_steps", 0), 2),
                round(d.get("elapsed_sec", 0), 1),
            ])
    print(f"\n[saved] {csv_path}")


if __name__ == "__main__":
    main()
