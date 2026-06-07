"""Solution A 보상: outcome rule reward = answer F1 + 작은 format 신호.

reward hacking 방어 (과거 교훈): F1을 주 신호로, format 보너스는 작게.
- 정답 파싱 실패 시 0 (parametric/포맷 붕괴 억제).
- 검색 한 번도 안 하고 답하면 format 보너스 제거 (검색 사용 유도).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..rewards import _normalize_answer, compute_answer_f1
from .format_utils import count_searches, parse_answer


@dataclass
class RewardBreakdown:
    total: float
    f1: float
    em: float
    fmt: float
    has_answer: bool
    n_search: int


def compute_em(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def compute_reward(
    rollout_text: str,
    gold: str,
    fmt_bonus: float = 0.1,
    require_search: bool = True,
) -> RewardBreakdown:
    """rollout_text = assistant가 생성한 전체 텍스트(주입된 information 포함 가능).

    reward = f1 + fmt_bonus * (well-formed answer & >=1 search)
    답 없으면 total=0.
    """
    pred = parse_answer(rollout_text)
    n_search = count_searches(rollout_text)
    if pred is None:
        return RewardBreakdown(0.0, 0.0, 0.0, 0.0, False, n_search)

    f1 = compute_answer_f1(pred, gold)
    em = compute_em(pred, gold)

    fmt = 0.0
    if fmt_bonus > 0.0:
        ok_search = (n_search >= 1) if require_search else True
        if ok_search:
            fmt = fmt_bonus

    total = f1 + fmt
    return RewardBreakdown(total, f1, em, fmt, True, n_search)
