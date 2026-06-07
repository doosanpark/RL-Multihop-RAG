"""gym 스타일 RAG 환경.

1 에피소드 = HotpotQA 1 샘플.
Action space는 고정 크기 2N+1 (N = 후보 단락 수, 기본 10).
이미 처리한 단락 action은 valid_actions_mask로 차단.

LLM 호출은 stop_and_answer 또는 max_steps / all-processed 도달 시에만.
LLM은 callable `answerer(question, kept_passages) -> str`로 추상화 →
실 학습/평가에선 Qwen 호출 함수를, 테스트에선 mock 함수를 주입.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rewards import (
    compute_answer_f1,
    compute_final_reward,
    compute_step_reward,
)
from .rl_types import Action, ActionKind, StepRecord, Trajectory


# answerer signature: (question, kept_passage_texts) -> predicted_answer_str
Answerer = Callable[[str, List[str]], str]


@dataclass
class Passage:
    title: str
    text: str           # context['sentences']를 join한 단락 본문
    sentences: List[str]
    is_gold: bool       # supporting_facts에 title이 포함되는가
    idx: int            # 원본 후보 list에서의 인덱스


@dataclass
class State:
    """env가 반환하는 raw state. 학습 시엔 별도 인코더로 벡터화."""

    question: str
    passages: List[Passage]
    processed: List[bool]
    kept_indices: List[int]
    step: int
    valid_actions_mask: List[bool]  # 길이 2N+1
    done: bool = False


class InvalidActionError(RuntimeError):
    pass


class RAGEnv:
    """HotpotQA 멀티홉 RAG를 MDP로 감싼 환경.

    Args:
        n_candidates: 후보 단락 수 (HotpotQA distractor split은 항상 10).
        max_steps: 한 에피소드 최대 step (기본 10).
        answerer: LLM 답변 함수. None이면 빈 문자열을 반환 (테스트용).
    """

    def __init__(
        self,
        n_candidates: int = 10,
        max_steps: int = 10,
        answerer: Optional[Answerer] = None,
    ) -> None:
        self.n_candidates = n_candidates
        self.max_steps = max_steps
        self.answerer = answerer

        # action space는 stop 액션 1개 + KEEP N개 + DROP N개 = 2N+1
        self.action_dim = 2 * n_candidates + 1

        # 에피소드 state (reset에서 초기화)
        self._sample: Optional[Dict[str, Any]] = None
        self._passages: List[Passage] = []
        self._kept_indices: List[int] = []
        self._processed: List[bool] = []
        self._step_count: int = 0
        self._gold_answer: str = ""
        self._question: str = ""
        self._done: bool = False
        self._trajectory: Trajectory = Trajectory()

    # ---------- 공개 API ----------

    def reset(self, sample: Dict[str, Any]) -> State:
        """1 HotpotQA 샘플을 받아 새 에피소드 시작."""
        titles = sample["context"]["title"]
        sentences_list = sample["context"]["sentences"]
        if len(titles) > self.n_candidates:
            # HotpotQA distractor는 보통 정확히 10이지만 안전장치
            titles = titles[: self.n_candidates]
            sentences_list = sentences_list[: self.n_candidates]

        gold_titles = set(sample["supporting_facts"]["title"])

        self._passages = []
        for i, (t, sents) in enumerate(zip(titles, sentences_list)):
            self._passages.append(
                Passage(
                    title=t,
                    text="".join(sents),
                    sentences=list(sents),
                    is_gold=(t in gold_titles),
                    idx=i,
                )
            )

        self._sample = sample
        self._kept_indices = []
        self._processed = [False] * len(self._passages)
        self._step_count = 0
        self._gold_answer = sample["answer"]
        self._question = sample["question"]
        self._done = False
        self._trajectory = Trajectory(gold_answer=self._gold_answer)

        return self._snapshot()

    def step(self, action_idx: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """한 step 진행.

        Returns:
            (next_state, reward, done, info)
            reward: step_reward + (done이면 final_reward 합산)
            info: action 메타데이터 + done일 때 answer_f1, predicted_answer 등
        """
        if self._done:
            raise InvalidActionError("episode is already terminated. call reset().")

        action = Action.from_index(action_idx, self.n_candidates)
        mask = self._valid_mask()
        if not mask[action_idx]:
            raise InvalidActionError(
                f"invalid action {action_idx} ({action.kind.value} idx="
                f"{action.passage_idx}). already-processed passages cannot "
                f"be re-acted on."
            )

        is_gold = False
        title = None
        if action.kind == ActionKind.KEEP:
            assert action.passage_idx is not None
            p = self._passages[action.passage_idx]
            self._kept_indices.append(action.passage_idx)
            self._processed[action.passage_idx] = True
            is_gold = p.is_gold
            title = p.title
        elif action.kind == ActionKind.DROP:
            assert action.passage_idx is not None
            p = self._passages[action.passage_idx]
            self._processed[action.passage_idx] = True
            is_gold = p.is_gold
            title = p.title
        # STOP은 상태 변화 없음

        r_step = compute_step_reward(action.kind, is_gold)
        self._step_count += 1
        self._trajectory.records.append(
            StepRecord(
                action=action,
                step_reward=r_step,
                is_gold=is_gold,
                passage_title=title,
            )
        )

        info: Dict[str, Any] = {
            "action_kind": action.kind.value,
            "passage_idx": action.passage_idx,
            "is_gold": is_gold,
            "step_reward": r_step,
        }

        reward = r_step
        done = (
            action.kind == ActionKind.STOP
            or self._step_count >= self.max_steps
            or all(self._processed)
        )

        if done:
            # 종료 시 LLM 호출 + final reward
            kept_texts = [self._passages[i].text for i in self._kept_indices]
            kept_titles = [self._passages[i].title for i in self._kept_indices]
            pred = self.answerer(self._question, kept_texts) if self.answerer else ""
            f1 = compute_answer_f1(pred, self._gold_answer)
            r_final = compute_final_reward(f1, self._step_count)

            self._trajectory.final_reward = r_final
            self._trajectory.answer_f1 = f1
            self._trajectory.predicted_answer = pred
            self._trajectory.kept_titles = kept_titles

            reward += r_final
            info.update(
                {
                    "final_reward": r_final,
                    "answer_f1": f1,
                    "predicted_answer": pred,
                    "kept_titles": kept_titles,
                    "total_steps": self._step_count,
                }
            )

        self._done = done
        return self._snapshot(), reward, done, info

    def render(self) -> str:
        """디버깅용 텍스트 출력. trajectory를 사람이 읽기 좋게 dump."""
        lines = [
            f"Q: {self._question}",
            f"Gold answer: {self._gold_answer}",
            f"Gold titles: {[p.title for p in self._passages if p.is_gold]}",
            f"Step count: {self._step_count} / max {self.max_steps}",
            "Trajectory:",
        ]
        for k, rec in enumerate(self._trajectory.records):
            gold_mark = "✓" if rec.is_gold else " "
            lines.append(
                f"  [{k+1:2d}] {rec.action.kind.value:>4} "
                f"idx={rec.action.passage_idx} gold={gold_mark} "
                f"title={rec.passage_title!r} r={rec.step_reward:+.3f}"
            )
        if self._done:
            lines += [
                f"Predicted: {self._trajectory.predicted_answer!r}",
                f"answer_F1={self._trajectory.answer_f1:.3f}",
                f"final_reward={self._trajectory.final_reward:+.3f}",
            ]
        return "\n".join(lines)

    # ---------- accessor ----------

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def passages(self) -> List[Passage]:
        return list(self._passages)

    @property
    def n_passages(self) -> int:
        return len(self._passages)

    @property
    def question(self) -> str:
        return self._question

    @property
    def gold_answer(self) -> str:
        return self._gold_answer

    @property
    def done(self) -> bool:
        return self._done

    # ---------- 내부 ----------

    def _valid_mask(self) -> List[bool]:
        """현재 step에서 유효한 action mask. 길이 2N+1.

        - 처리된 (kept 또는 dropped) 단락은 keep/drop 모두 불가.
        - STOP은 언제나 valid.
        """
        N = len(self._passages)
        mask = [False] * self.action_dim
        for i in range(N):
            if not self._processed[i]:
                mask[i] = True                       # keep_i
                mask[self.n_candidates + i] = True   # drop_i (오프셋은 항상 n_candidates 기준)
        # N < n_candidates인 샘플에서 존재하지 않는 단락 자리는 False 유지
        mask[2 * self.n_candidates] = True  # STOP
        return mask

    def _snapshot(self) -> State:
        return State(
            question=self._question,
            passages=list(self._passages),
            processed=list(self._processed),
            kept_indices=list(self._kept_indices),
            step=self._step_count,
            valid_actions_mask=self._valid_mask(),
            done=self._done,
        )


# ---------- 테스트용 mock answerer ----------


def make_oracle_answerer(gold_answer: str, gold_titles: List[str]) -> Answerer:
    """unit test용: kept 단락에 gold title이 포함되어 있으면 정답을, 아니면 빈 문자열.

    이 함수는 env 외부에서 sample마다 새로 만들어 주입한다. env가 정답을 직접
    들여다보지 않도록 분리하기 위함.
    """
    gold_set = set(gold_titles)

    def _answerer(question: str, kept_texts: List[str]) -> str:
        # kept_texts 자체엔 title이 없으므로 호출자가 gold_titles로 미리 묶어줘야 함.
        # 여기선 단순화: gold_titles 중 하나라도 kept_texts 안에 substring으로
        # 존재한다고 가정하지 않고, 호출자가 책임지는 방식. → 별도 oracle.
        raise NotImplementedError("use make_oracle_answerer_by_indices instead")

    return _answerer


def make_oracle_answerer_by_indices(
    passages: List[Passage], gold_answer: str
) -> Answerer:
    """단순한 oracle: kept 단락 중 하나라도 is_gold면 정답, 아니면 빈 문자열.

    실제 LLM이 아니므로 cross-domain 평가용으로는 부적합. 학습 흐름과 reward
    unit test 검증용으로만 사용.
    """
    # text 기준 매칭이 아닌 indices 기준으로 작동: env가 reset 후 호출자에게
    # passages를 줄 때, 그 passages 객체에 is_gold가 있으므로 closure에서 활용.
    gold_texts = {p.text for p in passages if p.is_gold}

    def _answerer(question: str, kept_texts: List[str]) -> str:
        if any(t in gold_texts for t in kept_texts):
            return gold_answer
        return ""

    return _answerer
