"""Search-R1 스타일 프로토콜 정의: 태그, 시스템 프롬프트, 프롬프트 빌더, 파서,
그리고 후보 풀 내 query-기반 retriever (MiniLM 재사용).

프로토콜 (한 assistant 턴 안에서 도구 결과가 inline 주입됨):
    <think>추론</think>
    <search>질의</search>            -> env가 <information>...</information> 주입
    <think>추론</think>
    <search>질의2</search>           -> <information>...</information>
    <think>추론</think>
    <answer>최종 답</answer>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---- 태그 상수 -------------------------------------------------------------
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SEARCH_OPEN, SEARCH_CLOSE = "<search>", "</search>"
INFO_OPEN, INFO_CLOSE = "<information>", "</information>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "You are a research assistant that answers multi-hop questions using a search tool.\n"
    "A fixed pool of candidate passages is available; the search tool retrieves from it.\n"
    "Follow this protocol exactly:\n"
    "- Put your reasoning between <think> and </think>.\n"
    "- To search, write <search>your query</search>. The system replies with results "
    "between <information> and </information>.\n"
    "- You may search more than once to gather facts across hops.\n"
    "- When you have enough evidence, give the final answer between <answer> and </answer>. "
    "Keep the answer short (a name, entity, or phrase)."
)


def build_prompt_messages(question: str) -> List[dict]:
    """chat 템플릿에 넣을 messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question.strip()}"},
    ]


def format_information(passages: Sequence[Tuple[str, str]]) -> str:
    """검색 결과 블록 텍스트. passages = [(title, text), ...]."""
    body = "\n".join(f"({i+1}) {title}: {text}" for i, (title, text) in enumerate(passages))
    return f"{INFO_OPEN}\n{body}\n{INFO_CLOSE}"


# ---- 파서 ------------------------------------------------------------------
_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_last_search(text: str) -> Optional[str]:
    """가장 마지막 <search>...</search> 질의. 없으면 None."""
    matches = _SEARCH_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def parse_answer(text: str) -> Optional[str]:
    """<answer>...</answer> 내용. 없으면 None."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def count_searches(text: str) -> int:
    return len(_SEARCH_RE.findall(text))


def has_wellformed_answer(text: str) -> bool:
    return parse_answer(text) is not None


# ---- Retriever -------------------------------------------------------------
@dataclass
class Retriever:
    """MiniLM 기반, 후보 풀 내 query top-k passage 검색.

    한 질문의 후보(보통 10개)를 set_pool로 미리 임베딩 → search로 top-k 반환.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    _model: object = None

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._pool_titles: List[str] = []
        self._pool_texts: List[str] = []
        self._pool_emb: Optional[np.ndarray] = None

    def set_pool(self, passages: Sequence[Tuple[str, str]]) -> None:
        self._pool_titles = [t for t, _ in passages]
        self._pool_texts = [x for _, x in passages]
        # title + text를 임베딩 대상으로 (title이 검색 신호로 강함)
        corpus = [f"{t}. {x}" for t, x in passages]
        self._pool_emb = self._model.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 2, max_words: int = 100) -> List[Tuple[str, str]]:
        assert self._pool_emb is not None, "set_pool 먼저 호출"
        q = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self._pool_emb @ q
        order = np.argsort(-sims)[:top_k]
        out = []
        for i in order:
            text = self._pool_texts[i]
            ws = text.split()
            if len(ws) > max_words:
                text = " ".join(ws[:max_words]) + " ..."
            out.append((self._pool_titles[i], text))
        return out
