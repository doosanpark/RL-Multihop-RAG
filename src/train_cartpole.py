"""CartPole-v1 sanity check.

마스터 컨텍스트 spec:
  - 통과 기준: 평균 reward >= 195 (최근 100 에피소드)
  - 최대 500 에피소드 안에 통과해야 함
  - 안 통과 → 학습 코드 자체에 버그. RAG 환경으로 가지 마세요

실행:
    python -m src.train_cartpole
    python -m src.train_cartpole --seed 42 --max-episodes 500
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from .agent import REINFORCEAgent, Transition


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode(agent: REINFORCEAgent, env, max_t: int = 1000):
    """1 에피소드 rollout. transitions와 누적 reward 반환."""
    obs, _ = env.reset()
    transitions: list[Transition] = []
    total_reward = 0.0
    for _ in range(max_t):
        state = torch.from_numpy(obs).float()
        action, log_prob = agent.select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        transitions.append(
            Transition(state=state, action=action, log_prob=log_prob, reward=float(reward))
        )
        total_reward += float(reward)
        if terminated or truncated:
            break
    return transitions, total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--lr-policy", type=float, default=1e-3)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target", type=float, default=195.0)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--no-normalize-adv", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        hidden_dims=(256, 128),
        grad_clip=1.0,
        normalize_advantage=not args.no_normalize_adv,
        device=args.device,
    )

    print(
        f"[setup] state_dim={state_dim}, n_actions={n_actions}, "
        f"seed={args.seed}, lr_p={args.lr_policy}, lr_v={args.lr_value}, "
        f"γ={args.gamma}, target={args.target}/{args.window}ep"
    )

    rewards_history = []
    window = deque(maxlen=args.window)
    solved_ep = None

    for ep in range(1, args.max_episodes + 1):
        transitions, total_reward = run_episode(agent, env)
        info = agent.update(transitions)
        rewards_history.append(total_reward)
        window.append(total_reward)

        if ep % 20 == 0 or ep == 1:
            avg = sum(window) / len(window)
            print(
                f"[ep {ep:4d}] R={total_reward:5.0f} avg{len(window):3d}={avg:6.2f} "
                f"p_loss={info['policy_loss']:+.3f} v_loss={info['value_loss']:.3f} "
                f"ent={info['entropy']:.3f}"
            )

        # 통과 조건
        if len(window) == args.window and sum(window) / len(window) >= args.target:
            solved_ep = ep
            avg = sum(window) / len(window)
            print(f"\n✓ SOLVED at episode {ep} (avg{args.window}={avg:.2f} >= {args.target})")
            break

    env.close()

    # 결과 저장
    out = {
        "solved": solved_ep is not None,
        "solved_episode": solved_ep,
        "max_episodes": args.max_episodes,
        "final_avg100": sum(window) / len(window) if window else None,
        "hyperparams": {
            "seed": args.seed,
            "lr_policy": args.lr_policy,
            "lr_value": args.lr_value,
            "gamma": args.gamma,
            "hidden_dims": [256, 128],
            "grad_clip": 1.0,
            "normalize_advantage": not args.no_normalize_adv,
        },
    }
    json_path = RESULTS_DIR / f"cartpole_seed{args.seed}.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.save(RESULTS_DIR / f"cartpole_rewards_seed{args.seed}.npy", np.array(rewards_history))
    print(f"\n[saved] {json_path}")

    # 그림은 별도 스크립트에서 (의존성 무거움 회피)
    if not solved_ep:
        print(
            f"\n✗ NOT SOLVED in {args.max_episodes} episodes. "
            f"final avg{args.window}={sum(window)/len(window):.2f}. "
            f"학습 코드에 버그가 있을 수 있음 — 트러블슈팅 §8-1 참고."
        )


if __name__ == "__main__":
    main()
