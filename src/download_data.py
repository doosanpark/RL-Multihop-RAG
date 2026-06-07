"""HotpotQA distractor split 다운로드 + 샘플 구조 확인 스크립트.

Phase 1 검증용. 실행:
    python -m src.download_data
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


# 프로젝트 루트 = 이 파일의 부모의 부모
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("[1/3] HotpotQA distractor split 로드 중...")
    # HF에 공식 미러: hotpot_qa (config: distractor)
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)

    print("\n[2/3] split별 샘플 수:")
    for split_name, split in ds.items():
        print(f"  - {split_name}: {len(split):,}")

    # 한 샘플 추출
    sample = ds["train"][0]
    print("\n[3/3] train[0] 구조:")
    print(f"  - keys: {list(sample.keys())}")
    print(f"  - question: {sample['question']}")
    print(f"  - answer: {sample['answer']}")
    print(f"  - type: {sample.get('type')}")
    print(f"  - level: {sample.get('level')}")

    # context 구조
    ctx = sample["context"]
    print(f"\n  context 타입: {type(ctx).__name__}")
    print(f"  context keys: {list(ctx.keys()) if hasattr(ctx, 'keys') else 'list'}")
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents_list = ctx.get("sentences", [])
        print(f"  context 단락 수: {len(titles)}")
        print(f"  첫 단락 title: {titles[0] if titles else 'N/A'}")
        if sents_list:
            first_sents = sents_list[0]
            print(f"  첫 단락 문장 수: {len(first_sents)}")
            preview = first_sents[0][:120] if first_sents else ""
            print(f"  첫 문장 미리보기: {preview}...")

    # supporting_facts 구조 — 핵심
    sf = sample["supporting_facts"]
    print(f"\n  supporting_facts 타입: {type(sf).__name__}")
    print(f"  supporting_facts keys: {list(sf.keys()) if hasattr(sf, 'keys') else 'list'}")
    if isinstance(sf, dict):
        sf_titles = sf.get("title", [])
        sf_sent_ids = sf.get("sent_id", [])
        print(f"  정답 fact 개수: {len(sf_titles)}")
        for t, sid in zip(sf_titles, sf_sent_ids):
            print(f"    - title='{t}', sent_id={sid}")

    # 첫 5개 train 샘플을 raw에 dump (디버깅용)
    dump_path = RAW_DIR / "hotpot_train_first5.json"
    with dump_path.open("w", encoding="utf-8") as f:
        json.dump([ds["train"][i] for i in range(5)], f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {dump_path}")
    print("\n완료. supporting_facts 포맷: dict(title=list[str], sent_id=list[int])")


if __name__ == "__main__":
    main()
