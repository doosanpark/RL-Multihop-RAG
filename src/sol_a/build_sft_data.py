"""supporting_facts + 정답으로 expert multi-hop 추론 trace 생성 (LoRA SFT용).

각 trace = 한 assistant 턴: <think><search>...</search> [<information> 주입] ... <answer>.
생성 토큰(think/search/answer)만 trainable, 주입된 <information>은 loss 마스킹.

실행:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.build_sft_data --n 4000 --out data/sol_a/sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .format_utils import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    INFO_CLOSE,
    INFO_OPEN,
    SEARCH_CLOSE,
    SEARCH_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    build_prompt_messages,
)
from .hotpot_data import Example, load_examples

ROOT = Path(__file__).resolve().parent.parent.parent

_STOP = {
    "the", "a", "an", "of", "is", "was", "were", "are", "in", "on", "at", "to",
    "for", "and", "or", "what", "which", "who", "whom", "whose", "where", "when",
    "how", "why", "did", "does", "do", "this", "that", "by", "with", "as", "from",
    "part", "city", "first",
}


def _words(s: str) -> List[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if w]


def order_gold_titles(question: str, gold_titles: List[str]) -> List[str]:
    """질문과 단어겹침(불용어 제거)이 큰 gold title을 먼저. stable sort."""
    qwords = {w for w in _words(question) if w not in _STOP}
    def score(t: str) -> int:
        return len({w for w in _words(t) if w not in _STOP} & qwords)
    return sorted(gold_titles, key=lambda t: -score(t))


def _truncate_words(text: str, max_words: int = 120) -> str:
    ws = text.split()
    if len(ws) <= max_words:
        return text
    return " ".join(ws[:max_words]) + " ..."


def _info_block(title: str, text: str) -> str:
    body = f"(1) {title}: {_truncate_words(text)}"
    return f"\n{INFO_OPEN}\n{body}\n{INFO_CLOSE}\n"


def build_trace(ex: Example) -> Dict | None:
    """한 Example -> SFT 레코드. gold 부족하면 None."""
    if not ex.gold_titles or not ex.answer:
        return None
    titles = order_gold_titles(ex.question, ex.gold_titles)[:3]
    text_by_title = dict(ex.gold_passages)

    segments: List[Dict] = []

    def gen(text: str) -> None:
        segments.append({"t": True, "text": text})

    def ctx(text: str) -> None:
        segments.append({"t": False, "text": text})

    is_cmp = ex.qtype == "comparison"
    n = len(titles)

    # ---- hop 1
    if is_cmp:
        intro = (
            f"{THINK_OPEN}This is a comparison question. I need facts about each entity. "
            f"Let me first look up {titles[0]}.{THINK_CLOSE}"
        )
    else:
        intro = (
            f"{THINK_OPEN}To answer this, I need to gather facts step by step. "
            f"First, let me search for {titles[0]}.{THINK_CLOSE}"
        )
    gen(intro + f"{SEARCH_OPEN}{titles[0]}{SEARCH_CLOSE}")
    ctx(_info_block(titles[0], text_by_title.get(titles[0], "")))

    # ---- hop 2..n
    for i in range(1, n):
        if is_cmp:
            think = f"{THINK_OPEN}Now let me look up {titles[i]} to compare.{THINK_CLOSE}"
        else:
            think = (
                f"{THINK_OPEN}From the previous result, I now need information about "
                f"{titles[i]}. Let me search for it.{THINK_CLOSE}"
            )
        gen(think + f"{SEARCH_OPEN}{titles[i]}{SEARCH_CLOSE}")
        ctx(_info_block(titles[i], text_by_title.get(titles[i], "")))

    # ---- 최종 추론 + 답
    if is_cmp:
        final = f"{THINK_OPEN}Comparing the retrieved facts, the answer is {ex.answer}.{THINK_CLOSE}"
    else:
        final = f"{THINK_OPEN}Combining the retrieved evidence, the answer is {ex.answer}.{THINK_CLOSE}"
    gen(final + f"{ANSWER_OPEN}{ex.answer}{ANSWER_CLOSE}")

    return {
        "id": ex.id,
        "question": ex.question,
        "answer": ex.answer,
        "qtype": ex.qtype,
        "level": ex.level,
        "gold_titles": ex.gold_titles,
        "messages": build_prompt_messages(ex.question),
        "segments": segments,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="data/sol_a/sft_train.jsonl")
    args = ap.parse_args()

    print(f"[load] {args.split} n={args.n} start={args.start}")
    examples = load_examples(args.split, n=args.n, start=args.start)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0
    type_counts: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            rec = build_trace(ex)
            if rec is None:
                n_skip += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            type_counts[ex.qtype] = type_counts.get(ex.qtype, 0) + 1

    print(f"[done] wrote {n_ok} traces (skipped {n_skip}) -> {out_path}")
    print(f"[types] {type_counts}")


if __name__ == "__main__":
    main()
