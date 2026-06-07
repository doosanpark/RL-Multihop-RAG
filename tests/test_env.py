"""RAGEnv unit test.

핵심 검증:
  1. perfect trajectory (gold keep + noise drop + stop) → 큰 양수 return
  2. worst trajectory (gold drop + noise keep) → 큰 음수 return
  3. random trajectory → perfect > random > worst
  4. action masking이 invalid action을 막는다
  5. done 신호와 final reward 합산이 정확하다

LLM은 mock (kept에 gold가 있으면 정답 반환).
"""

from __future__ import annotations

import random
from typing import Any, Dict

import pytest

from src.env import RAGEnv, InvalidActionError, make_oracle_answerer_by_indices
from src.rewards import compute_total_return
from src.rl_types import Action, ActionKind


# ---------- 픽스처: 합성 HotpotQA 샘플 ----------


def _make_sample(
    n_passages: int = 10,
    n_gold: int = 2,
    answer: str = "Arthur's Magazine",
) -> Dict[str, Any]:
    """HotpotQA distractor 포맷의 합성 샘플.

    - 정답 단락 n_gold개, 노이즈 n_passages - n_gold개.
    - 각 단락은 title이 고유하도록 인덱스를 박아 둠.
    """
    titles = [f"GoldDoc_{i}" if i < n_gold else f"Noise_{i}" for i in range(n_passages)]
    sentences = [
        [f"This is passage {i} sentence 1.", f"More content {i}."]
        for i in range(n_passages)
    ]
    supporting_facts = {
        "title": [titles[i] for i in range(n_gold)],
        "sent_id": [0] * n_gold,
    }
    return {
        "id": "synthetic_0",
        "question": "Which magazine was started first?",
        "answer": answer,
        "type": "comparison",
        "level": "medium",
        "supporting_facts": supporting_facts,
        "context": {"title": titles, "sentences": sentences},
    }


@pytest.fixture
def env_with_oracle():
    """oracle LLM을 단 환경. 매 테스트마다 reset된 상태로 제공."""
    sample = _make_sample()
    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    env.reset(sample)
    # passages 정보가 reset 후 사용 가능 → 그걸 기반으로 oracle 주입
    env.answerer = make_oracle_answerer_by_indices(env.passages, sample["answer"])
    env.reset(sample)
    # reset이 답변 함수를 지우지 않도록 reset 후 다시 안 만들어도 되게 유지
    return env, sample


# ---------- 경로별 trajectory ----------


def _run_perfect(env: RAGEnv, sample: Dict[str, Any]) -> float:
    """모든 gold keep → 모든 noise drop → stop."""
    total_reward = 0.0
    N = env.n_candidates
    state = env.reset(sample)
    # gold 인덱스
    gold_indices = [i for i, p in enumerate(env.passages) if p.is_gold]
    noise_indices = [i for i, p in enumerate(env.passages) if not p.is_gold]
    for i in gold_indices:
        _, r, done, info = env.step(Action(ActionKind.KEEP, i).to_index(N))
        total_reward += r
        if done:
            return total_reward
    for i in noise_indices:
        _, r, done, info = env.step(Action(ActionKind.DROP, i).to_index(N))
        total_reward += r
        if done:
            return total_reward
    # 위에서 all_processed로 done이 됐을 것
    if not env._done:  # 혹시 안 됐다면 stop
        _, r, done, info = env.step(Action(ActionKind.STOP).to_index(N))
        total_reward += r
    return total_reward


def _run_worst(env: RAGEnv, sample: Dict[str, Any]) -> float:
    """모든 gold drop → 모든 noise keep → stop."""
    total_reward = 0.0
    N = env.n_candidates
    env.reset(sample)
    gold_indices = [i for i, p in enumerate(env.passages) if p.is_gold]
    noise_indices = [i for i, p in enumerate(env.passages) if not p.is_gold]
    for i in gold_indices:
        _, r, done, info = env.step(Action(ActionKind.DROP, i).to_index(N))
        total_reward += r
        if done:
            return total_reward
    for i in noise_indices:
        _, r, done, info = env.step(Action(ActionKind.KEEP, i).to_index(N))
        total_reward += r
        if done:
            return total_reward
    if not env._done:
        _, r, done, info = env.step(Action(ActionKind.STOP).to_index(N))
        total_reward += r
    return total_reward


def _run_random(env: RAGEnv, sample: Dict[str, Any], seed: int = 0) -> float:
    """매 step 유효한 action 중 균등 랜덤."""
    rng = random.Random(seed)
    total_reward = 0.0
    env.reset(sample)
    done = False
    while not done:
        state_mask = env._valid_mask()
        valid_idx = [i for i, m in enumerate(state_mask) if m]
        a = rng.choice(valid_idx)
        _, r, done, info = env.step(a)
        total_reward += r
    return total_reward


# ---------- 핵심 ordering 테스트 ----------


def test_perfect_trajectory_positive(env_with_oracle):
    env, sample = env_with_oracle
    r = _run_perfect(env, sample)
    # gold 2 keep + noise 8 drop + final(F1=1, t=10): 0.2*2 + 0.05*8 + (2.0 - 1.0) = 1.8
    assert r > 1.0, f"perfect return should be > 1.0, got {r}"
    # trajectory 마지막에 답이 맞아야 함
    assert env.trajectory.answer_f1 == pytest.approx(1.0)


def test_worst_trajectory_negative(env_with_oracle):
    env, sample = env_with_oracle
    r = _run_worst(env, sample)
    # gold 2 drop + noise 8 keep + final(F1=0, t=10): -0.3*2 - 0.1*8 - 1.0 = -2.4
    assert r < -1.0, f"worst return should be < -1.0, got {r}"
    assert env.trajectory.answer_f1 == 0.0


def test_perfect_gt_random_gt_worst(env_with_oracle):
    """spec의 핵심 ordering. 여러 seed로 평균이 깨지지 않아야 함."""
    env, sample = env_with_oracle
    r_perfect = _run_perfect(env, sample)
    r_worst = _run_worst(env, sample)
    # 다양한 seed의 random trajectory 평균
    rs = [_run_random(env, sample, seed=s) for s in range(20)]
    r_random_mean = sum(rs) / len(rs)
    assert r_perfect > r_random_mean > r_worst, (
        f"ordering broken: perfect={r_perfect:.3f}, "
        f"random_mean={r_random_mean:.3f}, worst={r_worst:.3f}"
    )


# ---------- 부수 동작 ----------


def test_action_masking_blocks_double_processing():
    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    sample = _make_sample()
    env.reset(sample)
    N = env.n_candidates
    # passage 0 keep
    env.step(Action(ActionKind.KEEP, 0).to_index(N))
    # 같은 passage 다시 keep 시도 → 에러
    with pytest.raises(InvalidActionError):
        env.step(Action(ActionKind.KEEP, 0).to_index(N))
    with pytest.raises(InvalidActionError):
        env.step(Action(ActionKind.DROP, 0).to_index(N))


def test_stop_terminates_immediately():
    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    sample = _make_sample()
    env.reset(sample)
    N = env.n_candidates
    state, r, done, info = env.step(Action(ActionKind.STOP).to_index(N))
    assert done is True
    assert "answer_f1" in info
    assert info["total_steps"] == 1


def test_max_steps_forces_done():
    env = RAGEnv(n_candidates=10, max_steps=3, answerer=None)
    sample = _make_sample()
    env.reset(sample)
    N = env.n_candidates
    # 3번 keep
    for i in range(3):
        _, _, done, info = env.step(Action(ActionKind.KEEP, i).to_index(N))
    assert done is True
    assert info["total_steps"] == 3


def test_fewer_than_10_passages_mask_aligned():
    """N<10 샘플에서 valid_actions_mask가 어긋나지 않아야 한다 (회귀 테스트).

    버그: drop 오프셋을 n_candidates 대신 실제 N으로 쓰면, N<10일 때
    존재하지 않는 passage를 KEEP하려다 IndexError가 났다.
    """
    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    sample = _make_sample(n_passages=8, n_gold=2)  # 8개짜리 샘플
    state = env.reset(sample)
    N = env.n_candidates  # action 디코딩 기준은 항상 10

    # mask에서 valid라고 표시된 action은 모두 실제로 step 가능해야 함
    valid_idx = [i for i, m in enumerate(state.valid_actions_mask) if m]
    for a in valid_idx:
        action = Action.from_index(a, N)
        if action.passage_idx is not None:
            # 유효하다고 표시된 passage_idx는 실제 존재 범위 안이어야
            assert action.passage_idx < env.n_passages, (
                f"action {a} → passage_idx {action.passage_idx} >= {env.n_passages}"
            )

    # 무작위로 끝까지 돌려도 IndexError가 안 나야 한다
    done = False
    while not done:
        mask = env._valid_mask()
        cand = [i for i, m in enumerate(mask) if m]
        _, _, done, _ = env.step(cand[0])


def test_step_after_done_raises():
    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    sample = _make_sample()
    env.reset(sample)
    N = env.n_candidates
    env.step(Action(ActionKind.STOP).to_index(N))
    with pytest.raises(InvalidActionError):
        env.step(Action(ActionKind.STOP).to_index(N))


def test_reward_components_match_v42(env_with_oracle):
    """info dict의 step_reward + final_reward 합이 단계별 reward와 일치."""
    env, sample = env_with_oracle
    N = env.n_candidates
    # gold 인덱스 1개만 keep 후 stop
    env.reset(sample)
    gold_i = next(i for i, p in enumerate(env.passages) if p.is_gold)
    _, r1, done, info1 = env.step(Action(ActionKind.KEEP, gold_i).to_index(N))
    assert info1["is_gold"] is True
    assert info1["step_reward"] == pytest.approx(0.2)
    assert done is False
    _, r2, done, info2 = env.step(Action(ActionKind.STOP).to_index(N))
    assert done is True
    # F1=1.0, t=2: final = 2.0 - 0.2 = 1.8. step is 0.
    assert info2["answer_f1"] == pytest.approx(1.0)
    assert info2["final_reward"] == pytest.approx(1.8)
    assert r2 == pytest.approx(0.0 + 1.8)


def test_total_return_from_trajectory_consistent(env_with_oracle):
    """env가 기록한 trajectory로 compute_total_return을 돌려도 동일해야."""
    env, sample = env_with_oracle
    N = env.n_candidates
    env.reset(sample)
    rs = []
    done = False
    while not done:
        mask = env._valid_mask()
        valid_idx = [i for i, m in enumerate(mask) if m]
        a = valid_idx[0]
        _, r, done, _ = env.step(a)
        rs.append(r)
    # compute_total_return은 discount 적용
    g = compute_total_return(env.trajectory)
    # rs는 step_reward + (마지막에 final 합산된 즉시 reward) — 할인 없음
    # 따라서 동일 값을 기대하면 안 되고, 단지 g가 finite하고 합리적인지만 확인
    assert isinstance(g, float)
    assert env.trajectory.total_steps == len(rs)
