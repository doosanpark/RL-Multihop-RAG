"""HotpotQA distractor 샘플을 Solution A용 구조로 변환.

각 샘플 -> Example(question, answer, type, candidates[(title,text)], gold_titles,
gold_passages[(title,text)], gold_sentences_by_title).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

from datasets import load_dataset


@dataclass
class Example:
    id: str
    question: str
    answer: str
    qtype: str  # 'bridge' | 'comparison'
    level: str
    candidates: List[Tuple[str, str]]          # (title, full passage text) x ~10
    gold_titles: List[str]                     # distinct, supporting_facts 등장 순서
    gold_passages: List[Tuple[str, str]]       # (title, full text) gold만
    gold_sentences: Dict[str, List[str]]       # title -> 지지 문장 리스트


def _join_sentences(sents: List[str]) -> str:
    return " ".join(s.strip() for s in sents).strip()


def make_example(sample: dict) -> Example:
    ctx = sample["context"]
    titles: List[str] = ctx["title"]
    sents_list: List[List[str]] = ctx["sentences"]

    candidates = [(t, _join_sentences(s)) for t, s in zip(titles, sents_list)]
    title_to_text = dict(candidates)
    title_to_sents = {t: s for t, s in zip(titles, sents_list)}

    sf = sample["supporting_facts"]
    sf_titles: List[str] = sf["title"]
    sf_sids: List[int] = sf["sent_id"]

    # distinct gold titles, 첫 등장 순서 유지
    gold_titles: List[str] = []
    for t in sf_titles:
        if t not in gold_titles:
            gold_titles.append(t)

    gold_passages = [(t, title_to_text.get(t, "")) for t in gold_titles]

    gold_sentences: Dict[str, List[str]] = {t: [] for t in gold_titles}
    for t, sid in zip(sf_titles, sf_sids):
        sents = title_to_sents.get(t, [])
        if 0 <= sid < len(sents):
            gold_sentences[t].append(sents[sid].strip())

    return Example(
        id=sample["id"],
        question=sample["question"].strip(),
        answer=sample["answer"].strip(),
        qtype=sample.get("type", ""),
        level=sample.get("level", ""),
        candidates=candidates,
        gold_titles=gold_titles,
        gold_passages=gold_passages,
        gold_sentences=gold_sentences,
    )


@lru_cache(maxsize=2)
def _load_split(split: str):
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    return ds[split]


def load_examples(split: str = "train", n: int | None = None, start: int = 0) -> List[Example]:
    data = _load_split(split)
    end = len(data) if n is None else min(start + n, len(data))
    return [make_example(data[i]) for i in range(start, end)]
