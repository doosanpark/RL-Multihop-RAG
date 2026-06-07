"""CartPole 3-seed 학습 곡선을 results/cartpole_curve.png에 저장.

실행:
    python -m src.plot_cartpole
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def moving_avg(x: np.ndarray, w: int = 100) -> np.ndarray:
    if len(x) < w:
        return np.array([])
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> None:
    seeds = [42, 123, 7]
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed in seeds:
        rewards = np.load(RESULTS_DIR / f"cartpole_rewards_seed{seed}.npy")
        meta = json.loads((RESULTS_DIR / f"cartpole_seed{seed}.json").read_text())
        ax.plot(rewards, alpha=0.25, label=f"seed={seed} (raw)")
        ma = moving_avg(rewards, 100)
        if len(ma):
            x = np.arange(100, 100 + len(ma))
            ax.plot(x, ma, linewidth=2, label=f"seed={seed} avg100 (solved ep {meta['solved_episode']})")

    ax.axhline(195, color="black", linestyle="--", alpha=0.5, label="target 195")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.set_title("CartPole-v1 REINFORCE + learned baseline")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    out = RESULTS_DIR / "cartpole_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
