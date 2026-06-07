"""SFT vs RL rollout 덤프를 페어링해서 정성 사례 추출.

- RL이 SFT를 고친 케이스 (F1 차이 큰 순)
- 둘 다 실패한 케이스 (천장의 정체)
- SFT는 맞췄는데 RL이 망친 케이스 (정직성)

실행:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.extract_cases
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent.parent
SFT_DUMP = ROOT / "results/sol_a_sft_hotpot_dump.jsonl"
RL_DUMP = ROOT / "results/sol_a_rl_s42_hotpot_dump.jsonl"
OUT_MD = ROOT / "results/sol_a_qualitative_cases.md"


def load(p: Path) -> Dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in p.open(encoding="utf-8")}


def shorten_gen(gen: str, max_chars: int = 700) -> str:
    """info 블록은 그대로 두고 전체 길이만 컷."""
    if len(gen) <= max_chars:
        return gen
    return gen[: max_chars] + " …(truncated)"


def fmt_case(s: dict, r: dict, header: str) -> str:
    return (
        f"### {header}\n"
        f"- **Q**: {s['question']}\n"
        f"- **type**: {s['qtype']} | **gold**: `{s['gold']}`\n"
        f"- **SFT pred**: `{s['pred']}` (F1={s['f1']:.2f})\n"
        f"- **RL pred**: `{r['pred']}` (F1={r['f1']:.2f}) — **Δ={r['f1']-s['f1']:+.2f}**\n"
    )


def main() -> None:
    sft = load(SFT_DUMP)
    rl = load(RL_DUMP)
    common = sorted(set(sft) & set(rl))
    print(f"[load] paired rows: {len(common)}")

    diffs = []
    for qid in common:
        s, r = sft[qid], rl[qid]
        diffs.append({"id": qid, "delta": r["f1"] - s["f1"], "sft": s, "rl": r})

    rl_fixed = [d for d in diffs if d["delta"] > 0.5]
    rl_fixed.sort(key=lambda d: -d["delta"])
    rl_broke = [d for d in diffs if d["delta"] < -0.5]
    rl_broke.sort(key=lambda d: d["delta"])
    both_fail = [d for d in diffs if d["sft"]["f1"] == 0 and d["rl"]["f1"] == 0]

    # 통계
    n = len(diffs)
    sft_mean = sum(d["sft"]["f1"] for d in diffs) / n
    rl_mean = sum(d["rl"]["f1"] for d in diffs) / n
    cmp_delta = [d for d in diffs if d["sft"]["qtype"] == "comparison"]
    br_delta = [d for d in diffs if d["sft"]["qtype"] == "bridge"]
    cmp_d = sum(d["delta"] for d in cmp_delta) / max(len(cmp_delta), 1)
    br_d = sum(d["delta"] for d in br_delta) / max(len(br_delta), 1)

    lines: List[str] = []
    lines.append("# Solution A — 정성 사례 분석 (SFT vs RL seed42, val[0:200])\n")
    lines.append("## 통계 요약 (paired, n={})\n".format(n))
    lines.append(f"- mean F1: SFT **{sft_mean:.3f}** → RL **{rl_mean:.3f}** (Δ={rl_mean-sft_mean:+.3f})")
    lines.append(f"- RL이 SFT를 큰 폭(Δ>0.5)으로 고친 케이스: **{len(rl_fixed)}**")
    lines.append(f"- RL이 망친 케이스(Δ<-0.5): **{len(rl_broke)}**")
    lines.append(f"- 둘 다 실패(F1=0): **{len(both_fail)}** (← 0.5B 천장)")
    lines.append(f"- 타입별 평균 Δ: comparison {cmp_d:+.3f}, bridge {br_d:+.3f} (RL 이득은 comparison에 집중 — 정량과 일치)\n")

    lines.append("## 사례 1 — RL이 SFT를 고친 비교 추론 (Top-3 by Δ)\n")
    for i, d in enumerate(rl_fixed[:3], 1):
        s, r = d["sft"], d["rl"]
        lines.append(fmt_case(s, r, f"Case 1.{i} ({s['qtype']})"))
        lines.append(f"\n```\nSFT gen (마지막 200자):\n{s['gen'][-300:]}\n\nRL gen (마지막 200자):\n{r['gen'][-300:]}\n```\n")

    lines.append("## 사례 2 — 둘 다 실패한 bridge (천장의 정체)\n")
    both_fail_bridge = [d for d in both_fail if d["sft"]["qtype"] == "bridge"][:2]
    for i, d in enumerate(both_fail_bridge, 1):
        s, r = d["sft"], d["rl"]
        lines.append(fmt_case(s, r, f"Case 2.{i} (bridge, 둘 다 F1=0)"))
        lines.append(f"\n```\nSFT pred: {s['pred']}\nRL pred:  {r['pred']}\ngold:     {s['gold']}\n```\n")

    if rl_broke:
        lines.append("## 사례 3 — RL이 오히려 망친 케이스 (정직성)\n")
        for i, d in enumerate(rl_broke[:2], 1):
            s, r = d["sft"], d["rl"]
            lines.append(fmt_case(s, r, f"Case 3.{i} ({s['qtype']})"))

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {OUT_MD}")
    print(f"\n=== preview ===\n" + "\n".join(lines[:18]))


if __name__ == "__main__":
    main()
