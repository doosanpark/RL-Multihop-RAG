"""LoRA 어댑터 4개(SFT + RL s42/s123/s7)를 HuggingFace Hub에 업로드.

사용 전:
    1) huggingface-cli login   # 토큰 입력 (https://huggingface.co/settings/tokens)
    2) (선택) 환경변수 HF_USER 설정. 기본 'erid3232'.

실행:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.push_hub
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

ROOT = Path(__file__).resolve().parent.parent.parent
USER = os.environ.get("HF_USER", "erid3232")
PREFIX = "rag-rl-sol-a"   # repo 이름 접두

ADAPTERS = {
    "sft": ROOT / "models/sol_a/sft/best",
    "rl-s42": ROOT / "models/sol_a/rl_s42_v2/best",
    "rl-s123": ROOT / "models/sol_a/rl_s123_v2/best",
    "rl-s7": ROOT / "models/sol_a/rl_s7_v2/best",
}

CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- peft
- lora
- search-r1
- multi-hop-qa
- hotpot-qa
---

# {repo_name}

RAG_RL 클래스 프로젝트 — Solution A의 {tag} LoRA 어댑터.

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- 학습 방식: {desc}
- 데이터: HotpotQA distractor split
- 코드/보고서: https://github.com/{user}/RAG_RL  (자세한 결과는 `report_solution_a.md`)

## 사용
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="bfloat16").cuda()
model = PeftModel.from_pretrained(base, "{user}/{repo_name}")
```

## 평가 (HotpotQA val[200:400] n=200, held-out)
| 모델 | F1 | EM | comparison F1 |
|---|---|---|---|
| frozen-base | 0.006 | 0.000 | 0.007 |
| SFT | 0.434 | 0.340 | 0.428 |
| RL (3-seed mean) | 0.469±0.007 | 0.372±0.002 | 0.568±0.024 |
"""

DESC = {
    "sft": "Qwen LoRA를 supporting_facts 기반 멀티홉 추론 trace로 SFT warmup",
    "rl-s42": "SFT 위에 GRPO RL (lr 3e-5, KL 0.01, 100step, seed 42, dev-best)",
    "rl-s123": "SFT 위에 GRPO RL (seed 123, dev-best)",
    "rl-s7": "SFT 위에 GRPO RL (seed 7, dev-best)",
}


def main() -> None:
    info = whoami()
    print(f"[hf] logged in as: {info.get('name')}")
    api = HfApi()
    for tag, local in ADAPTERS.items():
        repo_name = f"{PREFIX}-{tag}"
        repo_id = f"{USER}/{repo_name}"
        print(f"\n[push] {local} -> {repo_id}")
        if not local.exists():
            print(f"  [skip] local missing: {local}")
            continue
        create_repo(repo_id, exist_ok=True, repo_type="model")
        # README 생성
        card = CARD_TEMPLATE.format(repo_name=repo_name, tag=tag, desc=DESC[tag], user=USER)
        (local / "README.md").write_text(card, encoding="utf-8")
        api.upload_folder(folder_path=str(local), repo_id=repo_id, repo_type="model")
        print(f"  -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
