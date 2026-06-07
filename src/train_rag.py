"""HotpotQA에서 REINFORCE+Baseline으로 Step-wise RL 학습.

마스터 컨텍스트 Phase 4. 학습 setup:
  - 학습 데이터: HotpotQA train 중 ~2000 샘플 (--n-train으로 조정)
  - Optimizer: Adam, lr=1e-4 (spec)
  - Discount γ = 0.95 (RAG에선 더 빨리 감소)
  - Batch: episode-by-episode
  - Gradient clipping max_norm=1.0
  - 3 seed 예정 (별도 호출)

W&B 로깅 (매 N step):
  - episode_reward (running mean)
  - answer_f1 (running mean)
  - policy_loss / value_loss / entropy
  - gold_keep_rate (정답 단락 keep 비율)
  - action_distribution

실행:
    python -m src.train_rag --seed 42 --n-episodes 5000 --use-llm
    python -m src.train_rag --seed 42 --n-episodes 50 --no-wandb --use-mock-llm  # smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset

from .agent import REINFORCEAgent, Transition
from .env import RAGEnv, make_oracle_answerer_by_indices
from .rewards import GAMMA
from .rl_types import ActionKind
from .state_encoder import (
    StateEncoder,
    encode_sample,
    expected_state_dim,
    state_to_vector,
)


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_answerer(args, env: RAGEnv):
    """args에 따라 LLM, mock, oracle 중 선택."""
    if args.use_mock_llm:
        # 항상 빈 문자열 (F1=0)
        return lambda q, kt: ""
    if args.use_oracle:
        # passage에 gold가 있으면 정답을 반환 (학습 디버깅용)
        # env.passages가 reset 후에야 채워지므로 reset 직후 호출자가 갱신해야 함
        return None  # 매 reset마다 main loop에서 새로 만든다
    if args.use_llm:
        from .llm import QwenAnswerer
        return QwenAnswerer(device=args.device)
    # 기본: mock (빈 문자열)
    return lambda q, kt: ""


def expert_action_index(state, passages, n_candidates: int) -> int:
    """BC용 expert 정책: 미처리 gold가 있으면 KEEP, 없으면 미처리 noise를 DROP,
    둘 다 없으면 STOP. (gold 먼저 keep → noise drop → stop)
    """
    from .rl_types import Action, ActionKind

    for i, p in enumerate(passages):
        if not state.processed[i] and p.is_gold:
            return Action(ActionKind.KEEP, i).to_index(n_candidates)
    for i, p in enumerate(passages):
        if not state.processed[i] and not p.is_gold:
            return Action(ActionKind.DROP, i).to_index(n_candidates)
    return Action(ActionKind.STOP).to_index(n_candidates)


def collect_bc_data(env, encoder, samples, n_candidates: int = 10):
    """expert를 env에서 굴려 (state_vec, action_idx, mask) 데이터 수집.

    LLM 호출 불필요 (reward 안 씀) → answerer를 빈 함수로 둠.
    """
    states, actions, masks = [], [], []
    env.answerer = lambda q, kt: ""  # BC엔 reward/답변 불필요
    for sample in samples:
        state = env.reset(sample)
        encoded = encode_sample(encoder, env.passages, env.question, n_candidates_max=n_candidates)
        done = False
        while not done:
            a = expert_action_index(state, env.passages, n_candidates)
            sv = state_to_vector(state, encoded, max_steps=env.max_steps)
            states.append(sv)
            actions.append(a)
            masks.append(state.valid_actions_mask)
            state, _, done, _ = env.step(a)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(masks, dtype=bool),
    )


def run_bc_warmup(agent, env, encoder, samples, epochs, batch_size,
                  bc_lr=1e-3, n_candidates=10):
    """behavior cloning으로 policy를 expert 행동에 미리 맞춤.

    BC 전용 lr(기본 1e-3, RL lr보다 큼)을 써서 충분히 수렴시킨 뒤
    원래 RL lr로 복원한다.
    """
    print(f"[bc] expert 데이터 수집 ({len(samples)} samples)...")
    states, actions, masks = collect_bc_data(env, encoder, samples, n_candidates)
    n = len(states)
    print(f"[bc] {n} transitions 수집. {epochs} epoch 학습 (bc_lr={bc_lr})...")

    # policy_opt lr을 BC용으로 임시 변경
    orig_lrs = [g["lr"] for g in agent.policy_opt.param_groups]
    for g in agent.policy_opt.param_groups:
        g["lr"] = bc_lr

    idx = np.arange(n)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        rng.shuffle(idx)
        losses = []
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            loss = agent.bc_update(
                torch.from_numpy(states[b]),
                torch.from_numpy(actions[b]),
                torch.from_numpy(masks[b]),
            )
            losses.append(loss)
        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"[bc] epoch {ep+1}/{epochs} ce_loss={np.mean(losses):.4f}")

    # 원래 lr 복원
    for g, lr in zip(agent.policy_opt.param_groups, orig_lrs):
        g["lr"] = lr

    # BC 직후 policy가 expert를 얼마나 따라하는지 train accuracy로 확인
    with torch.no_grad():
        st = torch.from_numpy(states).to(agent.device)
        mk = torch.from_numpy(masks).to(agent.device)
        logits = agent.policy(st, mk)
        pred = logits.argmax(dim=-1).cpu().numpy()
        acc = (pred == actions).mean()
    print(f"[bc] expert-imitation train accuracy = {acc:.3f}")


@torch.no_grad()
def eval_dev_f1(agent, env, encoder, dev_samples, answerer, n_candidates=10):
    """현재 정책(greedy)을 dev set에 굴려 평균 answer_F1 반환.

    drift가 있어도 best 시점을 잡기 위한 용도.
    answerer를 명시적으로 주입 (BC 데이터 수집이 env.answerer를 빈 함수로
    바꿔놓기 때문에 반드시 실제 LLM answerer로 복원해야 함).
    """
    f1s = []
    for sample in dev_samples:
        state = env.reset(sample)
        env.answerer = answerer  # 실제 LLM 강제 주입
        encoded = encode_sample(encoder, env.passages, env.question, n_candidates_max=n_candidates)
        done = False
        info = {}
        while not done:
            sv = state_to_vector(state, encoded, max_steps=env.max_steps)
            st = torch.from_numpy(sv).float()
            mk = torch.tensor(state.valid_actions_mask, dtype=torch.bool)
            a = agent.greedy_action(st, mk)
            state, _, done, info = env.step(a)
        f1s.append(info.get("answer_f1", 0.0))
    return float(np.mean(f1s)) if f1s else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=2000,
                        help="HotpotQA train에서 사용할 샘플 수")
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--lr-policy", type=float, default=1e-4)
    parser.add_argument("--lr-value", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-llm", action="store_true",
                        help="Qwen2.5-0.5B 답변 생성기 사용")
    parser.add_argument("--use-mock-llm", action="store_true",
                        help="LLM 안 쓰고 빈 답 반환 (smoke용)")
    parser.add_argument("--use-oracle", action="store_true",
                        help="kept에 gold 있으면 정답 반환 (학습 디버깅용)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="rag-rl")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--log-every", type=int, default=20,
                        help="콘솔 로그 / W&B 로그 간격 (episode 단위)")
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--use-step-reward", action="store_true", default=True,
                        help="step-wise reward 사용 (default True). Phase 5 sparse용으로 --no-step-reward")
    parser.add_argument("--no-step-reward", dest="use_step_reward", action="store_false")
    parser.add_argument("--batch-episodes", type=int, default=8,
                        help="이 개수만큼 에피소드를 모아 1회 업데이트 (variance 감소)")
    parser.add_argument("--bc-warmup-samples", type=int, default=300,
                        help="RL 전 behavior cloning warmup에 쓸 샘플 수 (0이면 비활성)")
    parser.add_argument("--bc-epochs", type=int, default=30,
                        help="BC warmup epoch 수")
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3,
                        help="BC 전용 학습률 (RL lr보다 크게)")
    parser.add_argument("--dev-eval-every", type=int, default=250,
                        help="dev F1 평가 간격 (episode). 0이면 비활성")
    parser.add_argument("--dev-n", type=int, default=50,
                        help="dev 평가 샘플 수 (validation split)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="optimizer L2 정규화 (과적합 억제)")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---------- W&B ----------
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            mode=args.wandb_mode,
            config=vars(args),
            name=f"step{'_off' if not args.use_step_reward else ''}_s{args.seed}_{int(time.time())}",
        )
    else:
        run = None

    # ---------- 데이터 ----------
    print(f"[data] HotpotQA train load ({args.n_train} samples)...")
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    train = ds["train"].shuffle(seed=args.seed).select(range(args.n_train))
    dev = None
    if args.dev_eval_every > 0:
        dev = ds["validation"].shuffle(seed=0).select(range(args.dev_n))
    print(f"[data] loaded {len(train)} train" + (f", {len(dev)} dev" if dev is not None else ""))

    # ---------- env / encoder / answerer ----------
    print(f"[encoder] loading sentence-transformers (device={args.device})...")
    encoder = StateEncoder(device=args.device)
    state_dim = expected_state_dim(n_candidates=10, emb_dim=encoder.emb_dim)
    n_actions = 2 * 10 + 1
    print(f"[env] state_dim={state_dim}, n_actions={n_actions}")

    env = RAGEnv(n_candidates=10, max_steps=10, answerer=None)
    answerer = build_answerer(args, env)

    # ---------- agent ----------
    agent = REINFORCEAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        hidden_dims=(256, 128),
        grad_clip=1.0,
        normalize_advantage=True,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    # ---------- BC warmup ----------
    if args.bc_warmup_samples > 0:
        bc_samples = [train[i] for i in range(min(args.bc_warmup_samples, len(train)))]
        run_bc_warmup(agent, env, encoder, bc_samples, args.bc_epochs,
                      args.bc_batch_size, bc_lr=args.bc_lr)

    # ---------- 학습 ----------
    window_R = deque(maxlen=100)
    window_F1 = deque(maxlen=100)
    window_gold_keep = deque(maxlen=100)
    action_counts = np.zeros(n_actions, dtype=np.int64)
    log_history: List[Dict[str, Any]] = []
    force_zero_step = not args.use_step_reward

    rng = random.Random(args.seed)
    t_start = time.time()

    episode_batch: List[List[Transition]] = []  # batch_episodes만큼 모아 업데이트
    update_info: Dict[str, Any] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    tag = "step" if args.use_step_reward else "sparse"
    best_dev_f1 = -1.0
    best_ckpt_path = MODELS_DIR / f"{tag}_seed{args.seed}_best.pt"

    # BC 직후 dev F1 (RL이 BC 대비 개선/악화하는지 기준선)
    if dev is not None:
        bc_dev_f1 = eval_dev_f1(agent, env, encoder, dev, answerer)
        print(f"[dev] BC 직후 dev_F1 = {bc_dev_f1:.4f}  (n={len(dev)})")
        best_dev_f1 = bc_dev_f1
        torch.save({"policy": agent.policy.state_dict(), "value": agent.value.state_dict(),
                    "args": vars(args), "ep": 0, "dev_f1": bc_dev_f1}, best_ckpt_path)

    for ep in range(1, args.n_episodes + 1):
        sample = train[rng.randrange(len(train))]
        state = env.reset(sample)

        if args.use_oracle:
            env.answerer = make_oracle_answerer_by_indices(env.passages, sample["answer"])
        else:
            env.answerer = answerer

        encoded = encode_sample(encoder, env.passages, env.question, n_candidates_max=10)

        transitions: List[Transition] = []
        ep_reward = 0.0
        ep_gold_keep = 0
        n_gold_passages = sum(1 for p in env.passages if p.is_gold)
        done = False
        info: Dict[str, Any] = {}
        while not done:
            sv = state_to_vector(state, encoded, max_steps=env.max_steps)
            state_t = torch.from_numpy(sv).float()
            mask_t = torch.tensor(state.valid_actions_mask, dtype=torch.bool)
            action_idx, log_prob = agent.select_action(state_t, mask_t)
            action_counts[action_idx] += 1

            next_state, reward, done, info = env.step(action_idx)
            if force_zero_step:
                # Sparse RL ablation: step reward 제거, final만 남김
                reward -= info.get("step_reward", 0.0)
            ep_reward += reward
            if info["action_kind"] == "keep" and info.get("is_gold"):
                ep_gold_keep += 1
            transitions.append(
                Transition(state=state_t, action=action_idx, log_prob=log_prob,
                           reward=float(reward), mask=mask_t)
            )
            state = next_state

        episode_batch.append(transitions)
        # batch_episodes만큼 모이면 업데이트
        if len(episode_batch) >= args.batch_episodes:
            update_info = agent.update_batch(episode_batch)
            episode_batch = []

        # logging
        f1 = info.get("answer_f1", 0.0) if info else 0.0
        gold_keep_rate = ep_gold_keep / max(1, n_gold_passages)
        window_R.append(ep_reward)
        window_F1.append(f1)
        window_gold_keep.append(gold_keep_rate)

        if ep % args.log_every == 0 or ep == 1:
            avg_R = sum(window_R) / len(window_R)
            avg_F1 = sum(window_F1) / len(window_F1)
            avg_gold = sum(window_gold_keep) / len(window_gold_keep)
            elapsed = time.time() - t_start
            ips = ep / max(1e-6, elapsed)
            print(
                f"[ep {ep:5d}/{args.n_episodes}] R={ep_reward:+.2f} avg={avg_R:+.2f} "
                f"F1={f1:.2f} avgF1={avg_F1:.2f} gold_keep={avg_gold:.2f} "
                f"p_loss={update_info['policy_loss']:+.3f} v_loss={update_info['value_loss']:.2f} "
                f"ent={update_info['entropy']:.2f} ips={ips:.1f}"
            )
            log_history.append(
                {
                    "episode": ep, "R": ep_reward, "avg_R": avg_R,
                    "F1": f1, "avg_F1": avg_F1, "gold_keep_rate": avg_gold,
                    **update_info,
                }
            )
            if run is not None:
                run.log(
                    {
                        "episode": ep,
                        "episode_reward": ep_reward,
                        "avg100/reward": avg_R,
                        "answer_f1": f1,
                        "avg100/f1": avg_F1,
                        "avg100/gold_keep_rate": avg_gold,
                        "policy_loss": update_info["policy_loss"],
                        "value_loss": update_info["value_loss"],
                        "entropy": update_info["entropy"],
                        "mean_return": update_info["mean_return"],
                        "elapsed_sec": elapsed,
                    },
                    step=ep,
                )

        # dev F1 평가 + best 체크포인트
        if dev is not None and ep % args.dev_eval_every == 0:
            dev_f1 = eval_dev_f1(agent, env, encoder, dev, answerer)
            star = ""
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                torch.save({"policy": agent.policy.state_dict(),
                            "value": agent.value.state_dict(),
                            "args": vars(args), "ep": ep, "dev_f1": dev_f1}, best_ckpt_path)
                star = " ★best"
            print(f"  [dev ep{ep}] dev_F1={dev_f1:.4f} (best={best_dev_f1:.4f}){star}")
            for h in log_history[::-1][:1]:
                h["dev_f1"] = dev_f1
            if run is not None:
                run.log({"dev_f1": dev_f1, "best_dev_f1": best_dev_f1}, step=ep)

        # checkpoint
        if ep % args.ckpt_every == 0:
            ckpt = MODELS_DIR / f"{tag}_seed{args.seed}_ep{ep}.pt"
            torch.save(
                {
                    "policy": agent.policy.state_dict(),
                    "value": agent.value.state_dict(),
                    "args": vars(args),
                    "ep": ep,
                },
                ckpt,
            )
            print(f"  [ckpt] {ckpt}")

    # 최종 저장
    tag = "step" if args.use_step_reward else "sparse"
    final_path = MODELS_DIR / f"{tag}_seed{args.seed}_final.pt"
    torch.save(
        {
            "policy": agent.policy.state_dict(),
            "value": agent.value.state_dict(),
            "args": vars(args),
            "ep": args.n_episodes,
            "action_counts": action_counts.tolist(),
        },
        final_path,
    )
    log_path = RESULTS_DIR / f"train_{tag}_seed{args.seed}.json"
    log_path.write_text(json.dumps(log_history, indent=2), encoding="utf-8")
    print(f"[done] final ckpt: {final_path}")
    print(f"[done] log: {log_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
