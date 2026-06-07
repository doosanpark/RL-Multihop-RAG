"""REINFORCE + learned baseline.

설계 원칙 (강의 06 범위 + variance reduction 표준 기법):
  - reward-to-go: G_t = r_t + γ G_{t+1}, 전체 return 사용 X
  - learned baseline V_φ(s): advantage A_t = G_t - V_φ(s_t)
  - advantage z-score 정규화: variance 더 줄이기
  - gradient clipping (max_norm=1.0)

Action masking을 지원해 RAGEnv에도 그대로 재사용 가능 (Phase 4).
mask는 bool tensor (shape: [n_actions])이며 True가 유효한 action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------- 네트워크 ----------


class PolicyNet(nn.Module):
    """2-layer MLP policy. state_dim → 256 → 128 → n_actions."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """logits 반환. mask가 주어지면 False 위치를 큰 음수로 채움.

        -inf 대신 -1e9를 쓰는 이유: BC cross-entropy / log_softmax에서 -inf가
        섞이면 NaN 위험이 있어 산술적으로 안전한 큰 음수를 사용.
        """
        logits = self.net(x)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return logits


class ValueNet(nn.Module):
    """동일 backbone + linear head로 scalar V(s) 추정."""

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,) shape으로 반환
        return self.net(x).squeeze(-1)


# ---------- 한 step에 저장할 정보 ----------


@dataclass
class Transition:
    """한 step 동안 학습에 필요한 텐서 / 값."""

    state: torch.Tensor                  # (state_dim,)
    action: int
    log_prob: torch.Tensor               # scalar (grad 유지)
    reward: float
    mask: Optional[torch.Tensor] = None  # (n_actions,) bool


# ---------- 에이전트 ----------


class REINFORCEAgent:
    """REINFORCE + learned baseline.

    Args:
        state_dim, n_actions: 환경별 크기
        lr_policy, lr_value: 별도 optimizer (서로 다른 학습률 가능)
        gamma: 할인율
        hidden_dims: policy/value 공통 hidden 구조
        grad_clip: gradient norm clipping (학습 안정성용)
        normalize_advantage: trajectory 내 advantage z-score 정규화 on/off
        device: "cpu" 또는 "cuda"
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr_policy: float = 1e-3,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        hidden_dims: Sequence[int] = (256, 128),
        grad_clip: float = 1.0,
        normalize_advantage: bool = True,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.normalize_advantage = normalize_advantage
        self.n_actions = n_actions

        self.policy = PolicyNet(state_dim, n_actions, hidden_dims).to(self.device)
        self.value = ValueNet(state_dim, hidden_dims).to(self.device)
        # weight_decay(L2)로 raw 임베딩 과적합 억제
        self.policy_opt = torch.optim.Adam(
            self.policy.parameters(), lr=lr_policy, weight_decay=weight_decay)
        self.value_opt = torch.optim.Adam(
            self.value.parameters(), lr=lr_value, weight_decay=weight_decay)

    # ----- action 선택 -----

    def select_action(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor]:
        """state로부터 action 샘플. (action_idx, log_prob) 반환.

        log_prob은 gradient를 유지한 채 반환됨 → trajectory에 모아두고
        update에서 직접 backward에 사용.
        """
        state_b = state.unsqueeze(0).to(self.device)
        mask_b = mask.unsqueeze(0).to(self.device) if mask is not None else None
        logits = self.policy(state_b, mask_b)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0)

    @torch.no_grad()
    def greedy_action(
        self,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> int:
        """평가용: argmax 선택. 학습에는 안 씀."""
        state_b = state.unsqueeze(0).to(self.device)
        mask_b = mask.unsqueeze(0).to(self.device) if mask is not None else None
        logits = self.policy(state_b, mask_b)
        return int(logits.argmax(dim=-1).item())

    # ----- update -----

    def update(self, transitions: List[Transition]) -> dict:
        """단일 trajectory 업데이트 (하위호환). 내부적으로 batch 1개로 위임."""
        return self.update_batch([transitions])

    def update_batch(self, episodes: List[List[Transition]]) -> dict:
        """여러 에피소드를 모아 한 번에 policy + value 업데이트.

        - reward-to-go는 에피소드별로 따로 계산 (경계 넘어 누적 X)
        - advantage 정규화는 배치 전체 transition 기준 → 짧은 에피소드에서도
          통계가 안정적 (단일 에피소드 정규화의 노이즈 증폭 문제 해결)
        - gradient는 배치 전체 평균 → variance 감소

        Returns: 로깅용 dict
        """
        # 비어있지 않은 에피소드만
        episodes = [ep for ep in episodes if len(ep) > 0]
        if not episodes:
            return {"policy_loss": 0.0, "value_loss": 0.0, "mean_return": 0.0,
                    "mean_advantage": 0.0, "entropy": 0.0}

        all_returns: List[torch.Tensor] = []
        all_states: List[torch.Tensor] = []
        all_log_probs: List[torch.Tensor] = []

        for transitions in episodes:
            T = len(transitions)
            returns = torch.zeros(T, device=self.device)
            running = 0.0
            for t in reversed(range(T)):
                running = transitions[t].reward + self.gamma * running
                returns[t] = running
            all_returns.append(returns)
            all_states.append(torch.stack([tr.state.to(self.device) for tr in transitions]))
            all_log_probs.append(torch.stack([tr.log_prob for tr in transitions]).to(self.device))

        returns = torch.cat(all_returns)        # (sum T,)
        states = torch.cat(all_states)          # (sum T, state_dim)
        log_probs = torch.cat(all_log_probs)    # (sum T,)
        n = returns.shape[0]

        # baseline
        values = self.value(states)
        advantages = returns - values.detach()

        if self.normalize_advantage and n > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)

        with torch.no_grad():
            logits = self.policy(states)
            entropy = Categorical(logits=logits).entropy().mean().item()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip)
        self.value_opt.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_return": float(returns.mean().item()),
            "mean_advantage": float(advantages.mean().item()),
            "entropy": entropy,
            "n_episodes": len(episodes),
        }

    def bc_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> float:
        """Behavior cloning: expert action을 지도학습 (cross-entropy).

        states (B, state_dim), actions (B,) long, masks (B, n_actions) bool.
        policy만 업데이트 (value는 RL 단계에서 학습).
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        masks = masks.to(self.device)
        logits = self.policy(states, masks)
        loss = F.cross_entropy(logits, actions)
        self.policy_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_opt.step()
        return float(loss.item())
