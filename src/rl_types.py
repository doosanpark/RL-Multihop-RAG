"""RL 환경에서 공유하는 데이터 타입.

Action space는 고정 크기 2*N + 1 (N=후보 단락 수).
  - idx in [0, N)        → KEEP passage idx
  - idx in [N, 2N)       → DROP passage (idx - N)
  - idx == 2N            → STOP_AND_ANSWER
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ActionKind(Enum):
    KEEP = "keep"
    DROP = "drop"
    STOP = "stop"


@dataclass
class Action:
    """Action space에서 한 행동의 의미 단위."""

    kind: ActionKind
    passage_idx: Optional[int] = None  # KEEP/DROP 일 때만 사용

    @classmethod
    def from_index(cls, idx: int, n_candidates: int) -> "Action":
        # 21차원 action space → (kind, passage_idx) 디코딩
        if idx == 2 * n_candidates:
            return cls(ActionKind.STOP, None)
        if 0 <= idx < n_candidates:
            return cls(ActionKind.KEEP, idx)
        if n_candidates <= idx < 2 * n_candidates:
            return cls(ActionKind.DROP, idx - n_candidates)
        raise ValueError(f"action index {idx} out of range for N={n_candidates}")

    def to_index(self, n_candidates: int) -> int:
        if self.kind == ActionKind.STOP:
            return 2 * n_candidates
        assert self.passage_idx is not None
        if self.kind == ActionKind.KEEP:
            return self.passage_idx
        return n_candidates + self.passage_idx


@dataclass
class StepRecord:
    """한 step의 결과. trajectory return 계산에 쓰임."""

    action: Action
    step_reward: float
    is_gold: bool          # 정답 단락이었는지 (KEEP/DROP 시에만 의미 있음)
    passage_title: Optional[str] = None  # 디버깅 / trajectory 시각화용


@dataclass
class Trajectory:
    """한 에피소드 전체. 학습 / 평가에서 모두 사용."""

    records: List[StepRecord] = field(default_factory=list)
    final_reward: float = 0.0
    answer_f1: float = 0.0
    predicted_answer: str = ""
    gold_answer: str = ""
    kept_titles: List[str] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.records)
