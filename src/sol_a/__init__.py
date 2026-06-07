"""Solution A: Qwen LoRA를 RL로 파인튜닝하여 검색+추론을 LLM 안에서 학습.

기존 selection-RL 프로젝트(폴백)와 분리. 재사용 자산:
  - src.rewards.compute_answer_f1  : answer F1 보상
  - sentence-transformers/all-MiniLM-L6-v2 : 후보 풀 내 query-기반 retrieve
  - HotpotQA distractor (HF on-the-fly)
"""
