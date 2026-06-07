"""Reward 함수 모음 — 마스터 컨텍스트 v4.2.

설계 원칙:
  - 비대칭: 정답 단락 drop은 -0.3, keep은 +0.2 (drop을 더 무겁게 페널티)
  - 연속 final: R_final = 2.0 * answer_F1 - 0.1 * t  (cliff 없음)
  - 균형: T=10일 때 step 누적 max ≈ 0.8 vs final max ≈ 2.0 → final 우세

답변 F1은 SQuAD 스타일 token-level F1.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import List

from .rl_types import ActionKind, StepRecord, Trajectory


# Step reward (비대칭, 마스터 컨텍스트 v4.2)
R_KEEP_GOLD = 0.2
R_DROP_GOLD = -0.3
R_KEEP_NOISE = -0.1
R_DROP_NOISE = 0.05
R_STOP = 0.0

# Final reward
FINAL_F1_COEF = 2.0
FINAL_STEP_PENALTY = 0.1

# Discount
GAMMA = 0.95


def compute_step_reward(action_kind: ActionKind, is_gold: bool) -> float:
    """단일 step의 즉시 보상.

    - is_gold: 해당 passage가 supporting fact에 포함된 정답 단락인지.
    - STOP은 0 (final reward가 처리).
    """
    if action_kind == ActionKind.STOP:
        return R_STOP
    if action_kind == ActionKind.KEEP:
        return R_KEEP_GOLD if is_gold else R_KEEP_NOISE
    if action_kind == ActionKind.DROP:
        return R_DROP_GOLD if is_gold else R_DROP_NOISE
    raise ValueError(f"unknown action_kind: {action_kind}")


def compute_final_reward(answer_f1: float, total_steps: int) -> float:
    """에피소드 종료 시 부여되는 답변 품질 보상.

    R_final = 2.0 * F1 − 0.1 * t
        - F1=1, t=5  → 1.5
        - F1=0, t=10 → -1.0
    """
    return FINAL_F1_COEF * answer_f1 - FINAL_STEP_PENALTY * total_steps


def compute_total_return(traj: Trajectory, gamma: float = GAMMA) -> float:
    """trajectory 한 개의 할인 누적 return.

    G_0 = Σ_{k=0..T-1} γ^k * r_step_k   +   γ^T * R_final
    """
    g = 0.0
    for k, rec in enumerate(traj.records):
        g += (gamma**k) * rec.step_reward
    g += (gamma ** len(traj.records)) * traj.final_reward
    return g


def compute_returns_to_go(
    traj: Trajectory, gamma: float = GAMMA
) -> List[float]:
    """REINFORCE에서 쓸 step별 G_t (reward-to-go).

    G_t = r_t + γ * r_{t+1} + ... + γ^(T-1-t) * r_{T-1} + γ^(T-t) * R_final
    """
    T = len(traj.records)
    rtg = [0.0] * T
    # 거꾸로 누적
    next_g = traj.final_reward
    for t in range(T - 1, -1, -1):
        rtg[t] = traj.records[t].step_reward + gamma * next_g
        next_g = rtg[t]
    return rtg


# ---------- Answer F1 (SQuAD-style) ----------


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def compute_answer_f1(pred: str, gold: str) -> float:
    """SQuAD 스타일 token-level F1.

    - 둘 다 공백/관사/문장부호 정규화 후 토큰 단위 비교.
    - 빈 문자열은 둘 다 비었을 때만 1.0, 그 외엔 0.0.
    """
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r)
