"""Solution A 결과 집계: 여러 eval JSON을 모아 모델별/seed별 mean±std 표 생성.

eval_a.py가 저장한 JSON들을 읽어 in-domain·transfer, hop별로 정리.

실행:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.aggregate_a
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RES = ROOT / "results"

# 집계 대상: (라벨, [eval json 경로들]) — 여러 개면 seed 평균
GROUPS = {
    "frozen-base (in-domain)": ["sol_a_frozen_heldout.json"],
    "SFT (in-domain)": ["sol_a_sft_heldout.json"],
    "RL 3-seed (in-domain)": [
        "sol_a_rl_s42_heldout.json",
        "sol_a_rl_s123_heldout.json",
        "sol_a_rl_s7_heldout.json",
    ],
    "frozen-base (sports)": ["sol_a_frozen_sports.json"],
    "SFT (sports)": ["sol_a_sft_sports.json"],
    "RL 3-seed (sports)": [
        "sol_a_rl_s42_sports.json",
        "sol_a_rl_s123_sports.json",
        "sol_a_rl_s7_sports.json",
    ],
}


def _get(d: Dict, key: str, field: str = "f1") -> float | None:
    if key in d and isinstance(d[key], dict) and d[key].get("n", 0) > 0:
        return d[key].get(field)
    return None


def _fmt(vals: List[float]) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  –  "
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def main() -> None:
    rows = []
    for label, files in GROUPS.items():
        loaded = []
        for f in files:
            p = RES / f
            if p.exists():
                loaded.append(json.loads(p.read_text(encoding="utf-8")))
        if not loaded:
            rows.append((label, "  –  ", "  –  ", "  –  ", "  –  ", "  –  "))
            continue
        overall = _fmt([_get(d, "overall", "f1") for d in loaded])
        em = _fmt([_get(d, "overall", "em") for d in loaded])
        bridge = _fmt([_get(d, "bridge", "f1") for d in loaded])
        comp = _fmt([_get(d, "comparison", "f1") for d in loaded])
        nseed = len(loaded)
        rows.append((label, overall, em, bridge, comp, f"{nseed}"))

    hdr = f"{'model':<28} {'F1':<14} {'EM':<14} {'bridge_F1':<14} {'comp_F1':<14} {'k':<3}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:<28} {r[1]:<14} {r[2]:<14} {r[3]:<14} {r[4]:<14} {r[5]:<3}")

    out = RES / "sol_a_summary.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump([dict(model=r[0], f1=r[1], em=r[2], bridge=r[3], comparison=r[4], k=r[5]) for r in rows],
                  f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
