"""멀티턴 search rollout 컨트롤러.

한 에피소드: prompt -> [generate until </search> -> retrieve -> inject <information>] * k
            -> generate until </answer>/eos.

policy gradient용으로 '모델이 생성한 토큰'만 action_mask=1, 주입된 information·프롬프트는 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from .format_utils import (
    ANSWER_CLOSE,
    SEARCH_CLOSE,
    build_prompt_messages,
    format_information,
    parse_answer,
    parse_last_search,
    Retriever,
)


@dataclass
class Rollout:
    question: str
    full_ids: List[int]         # prompt + (gen+info)*k + gen
    action_mask: List[int]      # len == full_ids, 1=모델 생성 토큰
    prompt_len: int
    gen_text: str               # 디코드된 assistant 전체 (info 포함)
    answer: Optional[str]
    n_search: int
    n_turns: int
    stop_reason: str            # 'answer' | 'eos' | 'max_turns' | 'no_stop'
    queries: List[str] = field(default_factory=list)


@dataclass
class RolloutConfig:
    max_turns: int = 3          # 최대 search 횟수
    max_new_tokens: int = 160   # 턴당 생성 상한
    max_total_tokens: int = 1100
    top_k: int = 2              # 검색 결과 passage 수
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True


@torch.no_grad()
def rollout_once(
    model,
    tok,
    retriever: Retriever,
    question: str,
    candidates: Sequence[Tuple[str, str]],
    cfg: RolloutConfig,
    device: str = "cuda",
) -> Rollout:
    retriever.set_pool(candidates)

    prompt_text = tok.apply_chat_template(
        build_prompt_messages(question), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids: List[int] = list(prompt_ids)
    action_mask: List[int] = [0] * len(prompt_ids)
    prompt_len = len(prompt_ids)

    queries: List[str] = []
    stop_reason = "no_stop"
    n_turns = 0

    for turn in range(cfg.max_turns + 1):
        if len(full_ids) >= cfg.max_total_tokens:
            stop_reason = "max_turns"
            break
        inp = torch.tensor([full_ids], device=device)
        budget = min(cfg.max_new_tokens, cfg.max_total_tokens - len(full_ids))
        out = model.generate(
            inp,
            max_new_tokens=budget,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            pad_token_id=tok.eos_token_id,
            stop_strings=[SEARCH_CLOSE, ANSWER_CLOSE],
            tokenizer=tok,
        )
        new_ids = out[0][len(full_ids):].tolist()
        # eos 제거 표시
        hit_eos = tok.eos_token_id in new_ids
        full_ids += new_ids
        action_mask += [1] * len(new_ids)
        new_text = tok.decode(new_ids, skip_special_tokens=True)
        n_turns = turn + 1

        if ANSWER_CLOSE in new_text:
            stop_reason = "answer"
            break
        if SEARCH_CLOSE in new_text:
            query = parse_last_search(new_text) or ""
            queries.append(query)
            if turn >= cfg.max_turns:  # 마지막 턴인데 또 search면 종료
                stop_reason = "max_turns"
                break
            passages = retriever.search(query, top_k=cfg.top_k) if query else []
            info_text = "\n" + format_information(passages) + "\n"
            info_ids = tok(info_text, add_special_tokens=False)["input_ids"]
            full_ids += info_ids
            action_mask += [0] * len(info_ids)
            continue
        if hit_eos:
            stop_reason = "eos"
            break
        # stop 없이 토큰 소진 -> 한 번 더 시도하지 않고 종료
        stop_reason = "no_stop"
        break

    gen_text = tok.decode(full_ids[prompt_len:], skip_special_tokens=True)
    answer = parse_answer(gen_text)
    return Rollout(
        question=question,
        full_ids=full_ids,
        action_mask=action_mask,
        prompt_len=prompt_len,
        gen_text=gen_text,
        answer=answer,
        n_search=len(queries),
        n_turns=n_turns,
        stop_reason=stop_reason,
        queries=queries,
    )
