"""State 인코더 — RAGEnv의 raw State를 MLP 입력 벡터로 변환.

벡터 구성 (n_candidates=10 기준):
  - question embedding (384,)
  - kept passages mean-pool (384,)  — kept가 비었으면 0 벡터
  - candidate passages stacked (10, 384) → flatten 3840
  - processed mask (10,) float — 이미 처리된 위치는 1
  - step (1,) normalized = step / max_steps
  - 총합 384 + 384 + 3840 + 10 + 1 = 4619차원

성능 메모: 한 샘플의 candidate / question 임베딩은 reset 시 1회 계산해
self._cache에 저장 → step마다 재인코딩 X.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .env import Passage, RAGEnv, State


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMB_DIM = 384


class StateEncoder:
    """SentenceTransformer 기반 인코더.

    Args:
        model_name: HF Hub의 sentence-transformers 모델 경로
        device: "cpu" 또는 "cuda"
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        # sentence-transformers 5.x에서 메서드 이름 변경
        if hasattr(self.model, "get_embedding_dimension"):
            self.emb_dim = self.model.get_embedding_dimension()
        else:
            self.emb_dim = self.model.get_sentence_embedding_dimension()
        assert self.emb_dim == DEFAULT_EMB_DIM, (
            f"unexpected emb_dim {self.emb_dim} (expected {DEFAULT_EMB_DIM})"
        )

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """배치 인코딩. shape (N, emb_dim) numpy float32."""
        if not texts:
            return np.zeros((0, self.emb_dim), dtype=np.float32)
        # convert_to_numpy=True가 약간 빠름. normalize는 cosine 유사도용이지만
        # 우리는 MLP 입력이라 raw 사용.
        emb = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


# ---------- 캐시 + state-to-vector ----------


class EncodedSample:
    """한 HotpotQA 샘플의 사전 인코딩 결과. env.reset 직후에 만들어 둠."""

    def __init__(
        self,
        question_emb: np.ndarray,
        passage_embs: np.ndarray,  # (N, emb_dim)
        n_candidates_max: int,
        emb_dim: int,
    ) -> None:
        self.question_emb = question_emb
        self.passage_embs = passage_embs
        self.n_candidates_max = n_candidates_max
        self.emb_dim = emb_dim
        # 단위벡터 (cosine 유사도 계산용 캐시)
        self._q_unit = question_emb / (np.linalg.norm(question_emb) + 1e-9)
        norms = np.linalg.norm(passage_embs, axis=1, keepdims=True)
        self._p_unit = passage_embs / (norms + 1e-9)  # (N, D)
        # question-passage cosine 유사도 (정적, 미리 계산)
        self.q_sim = (self._p_unit @ self._q_unit).astype(np.float32)  # (N,)


def encode_sample(
    encoder: StateEncoder,
    passages: List[Passage],
    question: str,
    n_candidates_max: int = 10,
) -> EncodedSample:
    """env.reset 직후 호출. question + passage 임베딩을 한 번에 계산."""
    q_emb = encoder.encode([question])[0]  # (D,)
    p_texts = [p.text for p in passages]
    p_embs = encoder.encode(p_texts)       # (N, D)
    # n_candidates_max에 맞춰 zero-pad
    if p_embs.shape[0] < n_candidates_max:
        pad = np.zeros((n_candidates_max - p_embs.shape[0], encoder.emb_dim), dtype=np.float32)
        p_embs = np.concatenate([p_embs, pad], axis=0)
    return EncodedSample(
        question_emb=q_emb,
        passage_embs=p_embs[:n_candidates_max],
        n_candidates_max=n_candidates_max,
        emb_dim=encoder.emb_dim,
    )


# state 표현 모드:
#   "lean" (기본): per-candidate 일반화 feature만 (q_sim, kept_sim, processed) + global.
#                  raw 임베딩 제거 → 암기 용량 없음 → cosine 수준으로 일반화.
#   "full": 원래의 raw 임베딩 포함 표현 (과적합 ablation용).
STATE_MODE = "lean"


def state_to_vector(
    state: State,
    encoded: EncodedSample,
    max_steps: int = 10,
    mode: str = STATE_MODE,
) -> np.ndarray:
    """RAGEnv State + 사전 인코딩 → MLP 입력 벡터 (1D)."""
    D = encoded.emb_dim
    N = encoded.n_candidates_max

    # kept mean-pool (없으면 0)
    if state.kept_indices:
        kept_pool = encoded.passage_embs[state.kept_indices].mean(axis=0)
    else:
        kept_pool = np.zeros(D, dtype=np.float32)

    # processed mask: 처리된 자리=1, 미처리=0, padding=1
    processed = np.ones(N, dtype=np.float32)
    n_real = len(state.passages)
    for i in range(n_real):
        processed[i] = 1.0 if state.processed[i] else 0.0

    step_norm = np.array([state.step / max(1, max_steps)], dtype=np.float32)

    # ----- 일반화 feature (핵심) -----
    # q_sim: question-passage cosine 유사도 (정적). cosine 베이스라인의 신호.
    q_sim = encoded.q_sim.copy()  # (N,)
    # kept_sim: 누적 keep 평균과의 cosine 유사도 → 중복/신규성. 적응적 선택용.
    kept_sim = np.zeros(N, dtype=np.float32)
    if state.kept_indices:
        kp_norm = kept_pool / (np.linalg.norm(kept_pool) + 1e-9)
        kept_sim = (encoded._p_unit @ kp_norm).astype(np.float32)  # (N,)

    if mode == "lean":
        # per-candidate 일반화 신호 + global. raw 임베딩 없음 → 암기 불가.
        n_kept = np.array([len(state.kept_indices) / max(1, N)], dtype=np.float32)
        vec = np.concatenate(
            [
                q_sim,        # N  질문 관련성 (cosine 신호)
                kept_sim,     # N  이미 keep한 것과의 중복도
                processed,    # N  처리 여부
                step_norm,    # 1
                n_kept,       # 1
            ],
            axis=0,
        )
    else:  # full (ablation)
        vec = np.concatenate(
            [
                encoded.question_emb,            # D
                kept_pool,                       # D
                encoded.passage_embs.flatten(),  # N*D
                processed,                       # N
                step_norm,                       # 1
                q_sim,                           # N
                kept_sim,                        # N
            ],
            axis=0,
        )
    return vec.astype(np.float32)


def expected_state_dim(
    n_candidates: int = 10, emb_dim: int = DEFAULT_EMB_DIM, mode: str = STATE_MODE
) -> int:
    """MLP 초기화 시 쓸 state vector 차원."""
    if mode == "lean":
        # q_sim + kept_sim + processed + step + n_kept
        return 3 * n_candidates + 2
    # full: question + kept_pool + candidates_flat + processed + step + q_sim + kept_sim
    return 2 * emb_dim + n_candidates * emb_dim + n_candidates + 1 + 2 * n_candidates
