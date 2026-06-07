"""Reward 함수 unit test.

목적: v4.2 spec에 명시된 비대칭성과 균형이 코드와 일치하는지 검증.
"""

from __future__ import annotations

import pytest

from src.rewards import (
    FINAL_F1_COEF,
    FINAL_STEP_PENALTY,
    R_DROP_GOLD,
    R_DROP_NOISE,
    R_KEEP_GOLD,
    R_KEEP_NOISE,
    R_STOP,
    compute_answer_f1,
    compute_final_reward,
    compute_returns_to_go,
    compute_step_reward,
    compute_total_return,
)
from src.rl_types import Action, ActionKind, StepRecord, Trajectory


# ---------- step reward 비대칭성 ----------


def test_step_reward_values_match_spec():
    # spec: keep_gold=+0.2, drop_gold=-0.3, keep_noise=-0.1, drop_noise=+0.05, stop=0
    assert compute_step_reward(ActionKind.KEEP, is_gold=True) == pytest.approx(0.2)
    assert compute_step_reward(ActionKind.DROP, is_gold=True) == pytest.approx(-0.3)
    assert compute_step_reward(ActionKind.KEEP, is_gold=False) == pytest.approx(-0.1)
    assert compute_step_reward(ActionKind.DROP, is_gold=False) == pytest.approx(0.05)
    assert compute_step_reward(ActionKind.STOP, is_gold=False) == pytest.approx(0.0)


def test_step_reward_drop_gold_is_most_negative():
    """비대칭 reward 핵심: 정답 단락 drop이 가장 큰 페널티."""
    rewards = [
        compute_step_reward(ActionKind.KEEP, True),
        compute_step_reward(ActionKind.KEEP, False),
        compute_step_reward(ActionKind.DROP, True),
        compute_step_reward(ActionKind.DROP, False),
    ]
    assert min(rewards) == compute_step_reward(ActionKind.DROP, True)


def test_drop_gold_more_punitive_than_keep_noise():
    # |drop_gold| > |keep_noise| 이어야 비대칭이 의미 있음
    assert abs(R_DROP_GOLD) > abs(R_KEEP_NOISE)


# ---------- final reward ----------


def test_final_reward_zero_f1_negative_penalty():
    # 답을 못 맞히면 step 페널티가 살아남아 음수
    assert compute_final_reward(answer_f1=0.0, total_steps=10) == pytest.approx(-1.0)


def test_final_reward_perfect_f1_dominates():
    # F1=1, t=5: 2.0 - 0.5 = 1.5
    assert compute_final_reward(1.0, 5) == pytest.approx(1.5)
    # F1=1, t=10: 2.0 - 1.0 = 1.0  ← 여전히 양수
    assert compute_final_reward(1.0, 10) == pytest.approx(1.0)


def test_final_dominates_step_at_max():
    """spec 균형 검증: F1=1로 받는 final이 step 누적 max보다 큼."""
    # step max는 keep_gold(0.2) 10번 = 2.0 (할인 무시 시), but T=10이면 final도 최대 2.0
    # 할인 적용 시 step 합 < 2.0이므로 final이 더 우세
    step_max_undiscounted = R_KEEP_GOLD * 10
    final_max = FINAL_F1_COEF * 1.0
    assert final_max >= step_max_undiscounted  # 동률 이상


# ---------- total return / reward-to-go ----------


def _make_traj(step_rewards: list[float], final: float) -> Trajectory:
    recs = [
        StepRecord(action=Action(ActionKind.STOP), step_reward=r, is_gold=False)
        for r in step_rewards
    ]
    return Trajectory(records=recs, final_reward=final)


def test_total_return_with_zero_step_is_discounted_final():
    traj = _make_traj([0.0, 0.0, 0.0], final=2.0)
    # G = γ^3 * 2.0
    expected = (0.95**3) * 2.0
    assert compute_total_return(traj) == pytest.approx(expected)


def test_total_return_pure_step_no_final():
    traj = _make_traj([1.0, 1.0, 1.0], final=0.0)
    # G = 1 + 0.95 + 0.95^2
    expected = 1.0 + 0.95 + 0.95**2
    assert compute_total_return(traj) == pytest.approx(expected)


def test_returns_to_go_satisfies_bellman():
    """G_t = r_t + γ * G_{t+1}, 마지막은 G_{T-1} = r_{T-1} + γ * final."""
    traj = _make_traj([0.5, 0.3, 0.2], final=2.0)
    rtg = compute_returns_to_go(traj)
    assert len(rtg) == 3
    # backward 검증
    assert rtg[2] == pytest.approx(0.2 + 0.95 * 2.0)
    assert rtg[1] == pytest.approx(0.3 + 0.95 * rtg[2])
    assert rtg[0] == pytest.approx(0.5 + 0.95 * rtg[1])


# ---------- answer F1 ----------


def test_f1_exact_match():
    assert compute_answer_f1("Arthur's Magazine", "Arthur's Magazine") == pytest.approx(1.0)


def test_f1_normalization_lowercase_and_articles():
    # 관사 / 대소문자 / 문장부호 정규화
    assert compute_answer_f1("The Arthur's Magazine.", "arthurs magazine") == pytest.approx(1.0)


def test_f1_no_overlap_zero():
    assert compute_answer_f1("foo bar", "baz qux") == 0.0


def test_f1_partial_overlap_between_zero_and_one():
    f1 = compute_answer_f1("New York City", "New York")
    assert 0.0 < f1 < 1.0


def test_f1_both_empty_is_one():
    # 둘 다 빈 답이면 1.0 (논쟁의 여지 있지만 SQuAD convention 따름)
    assert compute_answer_f1("", "") == 1.0


def test_f1_one_empty_is_zero():
    assert compute_answer_f1("", "Arthur") == 0.0
    assert compute_answer_f1("Arthur", "") == 0.0
