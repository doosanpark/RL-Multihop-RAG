"""실제 HotpotQA 샘플 1개로 env 동작 확인 — Phase 2 종료 sanity check.

실행:
    python -m src.smoke_env
"""

from __future__ import annotations

from datasets import load_dataset

from .env import RAGEnv, make_oracle_answerer_by_indices
from .rewards import compute_total_return
from .rl_types import Action, ActionKind


def main() -> None:
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    sample = ds["train"][0]

    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    state = env.reset(sample)
    print(f"Q: {state.question}")
    print(f"Gold answer: {sample['answer']}")
    print(f"Gold titles: {sample['supporting_facts']['title']}")
    print(f"Action dim: {env.action_dim} (=2*{env.n_candidates}+1)")
    print()

    # oracle 주입 후 perfect trajectory
    env.answerer = make_oracle_answerer_by_indices(env.passages, sample["answer"])
    env.reset(sample)
    env.answerer = make_oracle_answerer_by_indices(env.passages, sample["answer"])

    N = env.n_candidates
    gold = [i for i, p in enumerate(env.passages) if p.is_gold]
    noise = [i for i, p in enumerate(env.passages) if not p.is_gold]

    print(f"=== Perfect trajectory (keep gold {gold}, drop noise {noise}) ===")
    total = 0.0
    for i in gold:
        _, r, done, info = env.step(Action(ActionKind.KEEP, i).to_index(N))
        total += r
        print(f"  KEEP idx={i} title={env.passages[i].title:<32} r={r:+.3f} done={done}")
        if done:
            break
    if not env._done:
        for i in noise:
            _, r, done, info = env.step(Action(ActionKind.DROP, i).to_index(N))
            total += r
            print(f"  DROP idx={i} title={env.passages[i].title:<32} r={r:+.3f} done={done}")
            if done:
                break

    print(f"\nUndiscounted total reward = {total:+.3f}")
    print(f"Discounted G_0           = {compute_total_return(env.trajectory):+.3f}")
    print(f"answer_F1                 = {env.trajectory.answer_f1:.3f}")
    print(f"predicted answer          = {env.trajectory.predicted_answer!r}")
    print(f"kept titles               = {env.trajectory.kept_titles}")


if __name__ == "__main__":
    main()
