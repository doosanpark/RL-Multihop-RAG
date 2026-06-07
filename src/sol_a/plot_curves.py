"""3-seed RL의 dev F1 학습곡선 + SFT/cosine 기준선 플롯."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS = {"seed42": "rl_s42_v2", "seed123": "rl_s123_v2", "seed7": "rl_s7_v2"}
SFT_DEV = 0.415   # SFT dev[0:64]
COSINE = 0.370    # 기존 frozen+cosine in-domain (폴백)


def main() -> None:
    plt.figure(figsize=(7, 4.5))
    for label, d in SEEDS.items():
        h = json.loads((ROOT / "models/sol_a" / d / "history.json").read_text(encoding="utf-8"))
        xs = [r["step"] for r in h if "dev_f1" in r]
        ys = [r["dev_f1"] for r in h if "dev_f1" in r]
        plt.plot(xs, ys, marker="o", label=f"RL {label}")
    plt.axhline(SFT_DEV, ls="--", color="gray", label=f"SFT warmup ({SFT_DEV})")
    plt.axhline(COSINE, ls=":", color="black", label=f"prior cosine ({COSINE})")
    plt.xlabel("RL step")
    plt.ylabel("dev F1 (val[0:64], greedy)")
    plt.title("Solution A: GRPO RL dev F1 (3 seeds)\nimprove -> peak -> drift; dev-best preserves peak")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = ROOT / "results/sol_a_learning_curves.png"
    plt.savefig(out, dpi=130)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
