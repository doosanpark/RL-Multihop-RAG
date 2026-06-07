"""Qwen2.5-0.5B 기반 답변 함수.

설계:
  - LLM weight는 freeze (학습 안 함). 답변 생성 용도만.
  - greedy decoding (do_sample=False) → 학습 reproducibility 확보.
  - max_new_tokens 짧게 (50): HotpotQA 답변은 대부분 1-5 단어.
  - 컨텍스트가 비어도 (kept=[]) 호출 가능. 그땐 "no context" prompt로 빈 답이 잘 나오게.

성능:
  - 한 episode 종료 시 1회 호출 → 학습 병목.
  - bf16 + GPU + KV cache로 RTX 4060 Ti에서 ~0.5s/call 목표.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class QwenAnswerer:
    """Qwen2.5-Instruct를 RAG answerer로 감싸는 wrapper.

    호출 시그니처 (RAGEnv가 기대하는 형태):
        __call__(question: str, kept_texts: List[str]) -> str
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        # tokenizer / model 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device)
        self.model.eval()
        # 학습은 안 함 — gradient 차단
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _build_prompt(self, question: str, kept_texts: List[str]) -> str:
        if kept_texts:
            ctx = "\n\n".join(f"[Passage {i+1}]\n{t}" for i, t in enumerate(kept_texts))
        else:
            ctx = "(no passages selected)"
        # Qwen Instruct 채팅 템플릿 사용
        messages = [
            {
                "role": "system",
                "content": (
                    "You answer multi-hop questions based ONLY on the given "
                    "passages. Answer in as few words as possible (1-5 words). "
                    "If the passages are insufficient, answer 'unknown'."
                ),
            },
            {
                "role": "user",
                "content": f"Passages:\n{ctx}\n\nQuestion: {question}\nAnswer:",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def __call__(self, question: str, kept_texts: List[str]) -> str:
        prompt = self._build_prompt(question, kept_texts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(
            self.device
        )
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # 줄바꿈 / 마침표 등으로 잘리는 경우 첫 줄만
        text = text.split("\n")[0].strip()
        return text
