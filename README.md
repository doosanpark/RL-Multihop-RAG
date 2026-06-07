# RAG-RL: 강화학습 기반 Multi-hop QA 문서 선택

**서강대학교 강화학습 수업 프로젝트**
박두산 A72051 (팀장) · 신세정 A72058 (팀원)
GitHub: https://github.com/doosanpark/RL-Multihop-RAG.git

📊 **발표자료(보고서)**: [RL_Project_Presentation.pdf](RL_Project_Presentation.pdf) · [RL_Project_Presentation.pptx](RL_Project_Presentation.pptx)

---

## 프로젝트 개요

기존 RAG(Retrieval-Augmented Generation) 시스템은 질문과의 cosine 유사도 상위 k개 문서를 한 번에 선택해 LLM에 전달한다 (이하 **cosine 휴리스틱**, 본 보고서의 비교 기준선). 이 방식은 단순한 질문에는 잘 작동하지만, 여러 문서를 순서대로 참조해야 하는 **multi-hop 질문**에서는 중간 단계의 근거 문서를 놓치기 쉽다.

본 프로젝트는 이 문제를 **순차적 의사결정(Sequential Decision Making)**으로 재정의하고, 강화학습으로 문서 선택 정책을 학습하는 것을 목표로 한다. 실험은 두 단계로 나뉜다.

- **PART 1**: LLM은 고정(freeze)하고, 소형 MLP 정책망으로 문서 선택만 학습 (REINFORCE + Baseline)
- **PART 2**: 정책망의 한계를 진단한 뒤, LLM 자체를 LoRA SFT → GRPO RL로 파인튜닝하여 추론 능력까지 학습 (Search-R1 방식)

---

## 실험 결과 요약

### In-domain 성능 (HotpotQA validation, 3-seed mean ± std)

| 방법 | Answer F1 | 비고 |
|:--|--:|:--|
| Oracle (정답 문서만 입력) | 0.557 | Frozen Qwen2.5-0.5B의 이론적 상한 |
| **cosine 휴리스틱** (top-3) | 0.370 | 학습 없이 cosine 유사도로 top-3 선택 (Naive RAG) — PART 1이 넘지 못한 기준선 |
| **PART 1: Step-wise RL** | **0.355 ± 0.012** | H1/H2 기각 — cosine 휴리스틱에 미달 |
| PART 2: SFT search | 0.434 | 추론을 LLM 내부로 이전 → 기준선 돌파 |
| **PART 2: SFT + GRPO RL** | **0.469 ± 0.007** | RL 추가 이득 +0.035 (3-seed 견고) |
| Frozen base (cold-start, RL만) | 0.006 | SFT warmup 없이 RL만 적용 시 발산 → SFT 필요성 정량 입증 |

### 도메인 전이 성능 (스포츠 룰북 350문항, out-of-domain)

| 방법 | In-domain | Sports | 변화율 |
|:--|--:|--:|--:|
| **cosine 휴리스틱** (top-3) | 0.370 | 0.386 | **+4%** (견고) |
| PART 1: Step-wise RL | 0.355 | 0.270 ± 0.038 | **−24%** (H3 기각) |
| PART 2: SFT + GRPO RL | 0.469 | 0.313 ± 0.023 | **−33%** (과적합, 단 RL > SFT) |

**핵심 통찰**: 학습 없는 **cosine 휴리스틱**은 in-domain 성능 상한이 낮지만 OOD(Out-of-Distribution)에 견고하다. 학습된 정책은 in-domain에서 성능이 오르지만 OOD에 취약해진다. 두 PART 모두에서 **학습 효과 vs OOD 강건성의 트레이드오프**가 정량적으로 확인된다.

---

## 가설 검증 결과

### PART 1 (Selection-only RL)

| 가설 | 결과 | 근거 |
|:--|:--|:--|
| H1: Step-wise reward > Sparse reward | **기각** | 3-seed answer F1 동률 (0.355 vs 0.354). Step의 우위는 단일 seed noise였음. 단, support_F1 분산은 step에서 더 작음 (±0.005 vs ±0.027) |
| H2: RL > cosine 휴리스틱 | **기각** | 0.355 < 0.370. 단, RL은 2.0개 문서만 keep해 cosine 휴리스틱(3.0개)보다 간결하게 유사한 F1 달성 |
| H3: HotpotQA → 새 도메인 전이 | **기각** | Sports -24%, 랜덤 수준까지 하락 |

### PART 2 (SFT + GRPO RL)

| 가설 | 결과 | 근거 |
|:--|:--|:--|
| HA-1: 추론을 LLM 안에 두면 cosine 휴리스틱·selection-RL을 넘는다 | **강하게 지지** | 0.355 → 0.434 → 0.469. Cold-start 0.006이 SFT 기여를 정량 입증 |
| HA-2: RL이 SFT 위에 추가 이득을 준다 | **부분 지지** | +0.035, std 0.007로 견고. Comparison 타입에서 특히 두드러짐 |
| HA-3: 파인튜닝된 정책의 도메인 전이 | **부정적/혼합** | Sports 0.313 < cosine 휴리스틱 0.386. 단 RL > SFT이고, comparison 전이는 0.507로 강함 |

---

## PART 1 — Selection-only RL

### 문제 정의 (MDP 정식화)

1 에피소드 = HotpotQA 샘플 1개 (후보 문서 N=10: gold 2개 + distractor 8개).
매 step에서 agent가 문서 하나를 선택하고, 에피소드는 `stop_and_answer` 액션에서만 종료된다.

| 구성 요소 | 정의 |
|:--|:--|
| **State** | 질문 + 누적 keep 문서 + 후보별 상태 (q_sim, kept_sim, 처리 여부, step t) |
| **Action** | `keep pᵢ` / `drop pᵢ` / `stop_and_answer` (크기 2N+1, 처리된 문서는 mask 처리) |
| **Reward** | Step reward (즉시) + Final reward (종료 시): `R_final = 2.0 × answer_F1 − 0.1·t`, γ = 0.99 |
| **Policy** | 2-layer MLP (~200K params), LLM은 freeze하고 답변 생성에만 사용 |
| **Encoder** | sentence-transformers MiniLM-L6-v2 (384d, freeze) |
| **Algorithm** | REINFORCE + Learned Baseline (강의 06 범위) |

### Step Reward 설계 (v4.2)

```
keep  + 정답 문서  → +0.20
drop  + 정답 문서  → −0.30  (recall 보존을 위한 최대 페널티)
keep  + 노이즈 문서 → −0.10
drop  + 노이즈 문서 → +0.05
stop_and_answer    →  0 (Final reward로 이행)
```

### 주요 설계 결정 및 근거

**1. BC (Behavioral Cloning) warmup**
Cold-start 시 정책이 "즉시 stop" 국소 최적에 수렴하는 현상이 관측됨.
빈 컨텍스트 F1 = 0.149이고 이때 reward = 2×0.149 − 0.1 = +0.198으로, 관측 평균과 정확히 일치 → **reward hacking 정량 포착**.
`supporting_facts` 기반 expert 시연 1,000개로 모방학습(BC) 후 RL을 시작해 이 국소 최적을 회피.

**2. Lean state (32d)**
Raw 임베딩(4,639d) 입력 시 훈련 과적합 발생 (dev F1 0.19 < cosine 휴리스틱 0.37).
cosine 유사도 기반 lean state(q_sim, kept_sim, 처리 여부, step)로 차원을 줄이자 일반화 회복 (dev 0.35).

**3. CartPole sanity check**
본 학습 전 CartPole-v1에서 3-seed 검증 완료 (`avg(100ep) ≥ 195`).

| Seed | Solved episode | Final avg100 |
|--:|--:|--:|
| 42 | 242 | 196.66 |
| 123 | 185 | 196.56 |
| 7 | 157 | 195.47 |

---

## PART 2 — Solution A: LLM 내부에 추론 학습 (SFT → GRPO)

### PART 1의 진단과 해결 방향

PART 1의 실패 원인: **정책망(MLP)이 multi-hop 추론 용량을 갖추지 못함** (frozen LLM은 문서가 바르게 선택돼도 답을 제대로 못 냄 — oracle도 F1 0.557에 그침).

해결: 선택기와 추론기를 분리하는 대신, **LLM 자체가 검색-추론 전 과정을 한 assistant 턴 내에서 수행하도록** 학습.

### 학습 프로토콜

LLM (Qwen2.5-0.5B-Instruct)을 LoRA로 파인튜닝.
프로토콜은 `<think>` → `<search>` → env의 `<information>` 주입 → (반복) → `<answer>` 구조를 따름.
검색은 외부 서버 없이 후보 풀(10개) 내 MiniLM top-2 retrieve로 한정.

**Step 1: SFT warmup** (4,000 trace, 3 epoch, ~30분)
`supporting_facts`로 자동 생성한 멀티홉 추론 trace로 기본 포맷을 학습.

**Step 2: GRPO RL** (lr 3e-5, KL coef 0.01, 100 step, ~5h/seed)
GRPO = REINFORCE + **group baseline** (질문당 G=5 rollout의 평균) + **KL 정규화**
→ 강의 06의 baseline / variance reduction 개념의 자연스러운 확장.

### 상세 결과

| 모델 | In-domain F1 | Bridge | Comparison | Sports (OOD) |
|:--|--:|--:|--:|--:|
| Frozen base (cold-start) | 0.006 | 0.005 | 0.007 | 0.005 |
| SFT search | 0.434 | 0.435 | 0.428 | 0.299 |
| **SFT + GRPO RL (3-seed)** | **0.469 ± 0.007** | 0.445 ± 0.010 | **0.568 ± 0.024** | 0.313 ± 0.023 |

**학습 동역학**: improve → peak → drift 패턴. Seed 7은 step 100에서 포맷 붕괴(dev 0.354, search 횟수 1.45).
Dev-best 체크포인트 저장으로 peak 보존 (PART 1 교훈 재적용).

**정성 사례**: 동일한 검색 결과를 받고도 SFT는 등장인물("Kinsey Millhone")을, RL은 정답("C. W. Grafton")을 출력.
→ RL이 학습한 것은 새로운 검색 능력이 아니라, **질문 의도와 답 토큰 간 정렬의 정교화**.

---

## 환경 설정 및 실행

### 사전 요구사항

- OS: Windows / Linux
- GPU: NVIDIA 8GB 이상 (CUDA 12.1)
- Python: 3.11 (3.14는 PyTorch wheel 미지원)

### 설치

```powershell
# 1. 가상환경 생성 및 활성화
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1

# 2. PyTorch (CUDA 12.1) 설치
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 나머지 의존성 설치
pip install -r requirements.txt
```

> **주의**: 시스템 Python이 3.14로 설정된 경우 `datasets`/`dill` pickle 충돌이 발생한다.
> 항상 venv의 Python을 사용할 것. VS Code에서는 `.vscode/settings.json`으로 자동 활성화됨.

### PART 1 실행

```powershell
# CartPole sanity check (본 학습 전 필수)
python -m src.train_cartpole --seed 42 --max-episodes 500

# Baseline 평가 (학습 전)
python -m src.run_eval --variant top_k_sim --k 3 --n 200   # Naive RAG
python -m src.run_eval --variant oracle    --n 200          # 상한
python -m src.run_eval --variant random    --k 3 --n 200    # 하한

# 본 학습 (3-seed)
foreach ($s in 42,123,7) {
  python -m src.train_rag --seed $s --n-episodes 2500 --use-llm --no-wandb `
    --lr-policy 1e-4 --gamma 0.99 --bc-warmup-samples 1000 --batch-episodes 8 `
    --dev-eval-every 250 --weight-decay 1e-4         # step-wise reward
  python -m src.train_rag --seed $s --no-step-reward  # sparse reward (ablation)
}

# 평가 및 결과 집계
python -m src.run_eval --variant rl --ckpt models/step_seed42_best.pt --n 200
python -m src.aggregate_results   # results/table1_3seed.json + learning_curves.png
```

### PART 2 실행

```powershell
# SFT warmup (~30분)
python -m src.sol_a.sft_train --epochs 3

# GRPO RL (seed당 ~5시간, 중단 시 --resume models/sol_a/rl_s42_v2/ckpt 로 재개)
foreach ($s in 42,123,7) {
  python -m src.sol_a.grpo_train --steps 100 --lr 3e-5 --kl-coef 0.01 --seed $s `
    --out models/sol_a/rl_s${s}_v2
}

# 평가 (held-out: val[200:400], dev[0:64]와 disjoint)
python -m src.sol_a.eval_a --adapter models/sol_a/rl_s42_v2/best --dataset hotpot --n 200 --start 200
python -m src.sol_a.eval_a --adapter models/sol_a/rl_s42_v2/best --dataset sports
python -m src.sol_a.aggregate_a   # 3-seed 평균 ± 표준편차 집계
```

### 대화형 데모

```powershell
# HotpotQA 샘플로 질문하기
python -m src.ask

# 직접 문서를 입력해 테스트
python -m src.ask --mode passages

# 학습된 RL 정책으로 문서 선택
python -m src.ask --policy rl --ckpt models/step_seed42_final.pt
```

---

## 학습된 모델 다운로드

모든 체크포인트를 단일 zip (~40 MB)으로 제공.

**[model_releases/rag_rl_checkpoints.zip](https://github.com/doosanpark/RL-Multihop-RAG/raw/main/model_releases/rag_rl_checkpoints.zip)**

| 단계 | 경로 | 형식 | 크기 |
|:--|:--|:--|:--|
| PART 2 SFT (LoRA warmup) | `part2_sft/` | PEFT LoRA adapter | 8.6 MB |
| PART 2 RL seed 42 (dev-best) | `part2_rl_s42/` | PEFT LoRA adapter + history.json | 8.6 MB |
| PART 2 RL seed 123 (dev-best) | `part2_rl_s123/` | 동일 | 8.6 MB |
| PART 2 RL seed 7 (dev-best) | `part2_rl_s7/` | 동일 | 8.6 MB |
| PART 1 Step-wise RL (seed 42/123/7) | `part1_step_seed{42,123,7}_best.pt` | PyTorch state_dict | ~1 MB × 3 |
| PART 1 Sparse RL (seed 42/123/7, ablation) | `part1_sparse_seed{42,123,7}_best.pt` | 동일 | ~1 MB × 3 |

```python
# PART 2 RL 어댑터 로드
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).cuda()
model = PeftModel.from_pretrained(base, "part2_rl_s42")
```

```python
# PART 1 정책망 로드
import torch
ckpt = torch.load("part1_step_seed42_best.pt", weights_only=False)
# 상세 키는 src/agent.py 참고
```

---

## 프로젝트 구조

```
src/
  # PART 1 — Selection-only RL
  rl_types.py          Action / StepRecord / Trajectory 타입 정의
  rewards.py           v4.2 reward 함수 + answer F1 계산
  env.py               RAGEnv (gym 스타일 MDP)
  agent.py             REINFORCE + learned baseline
  state_encoder.py     sentence-transformers 기반 lean state(32d) 인코딩
  llm.py               Qwen2.5-0.5B 답변 생성 (freeze)
  train_cartpole.py    CartPole sanity check
  train_rag.py         HotpotQA 본 학습 (step-wise / sparse)
  evaluate.py          범용 평가기
  run_eval.py          통합 평가 엔트리포인트
  demo.py              단일 샘플 시각화 데모
  ask.py               대화형 질문 인터페이스
  baselines/
    naive_rag.py       use_all / top_k_sim / random / oracle
  # PART 2 — Solution A (Search-R1 SFT → GRPO)
  sol_a/
    format_utils.py    프로토콜 태그 · 파서 · MiniLM 검색기
    hotpot_data.py     HotpotQA 구조화 로더
    build_sft_data.py  supporting_facts → 멀티홉 SFT trace 생성
    sft_train.py       LoRA SFT warmup
    search_env.py      멀티턴 search rollout (</search> 정지 + 결과 주입)
    reward_a.py        outcome F1 + format 보상
    grpo_train.py      GRPO 학습 (KL + group baseline, --resume 지원)
    eval_a.py          in-domain / sports 평가
    aggregate_a.py     3-seed 결과 집계
    extract_cases.py   정성 사례 추출 (SFT vs RL paired)
    plot_curves.py     dev F1 학습 곡선 plot
    make_release.py    모델 zip 빌드
    push_hub.py        HuggingFace Hub 업로드 스크립트
    md_to_docx.py      markdown 보고서 → .docx 변환

data/
  raw/                 원본 HotpotQA 캐시 (gitignored)
  processed/           임베딩 캐시
  eval/                sports.json (스포츠 룰북 350문항, OOD 전이 평가셋)
  sol_a/               PART 2 SFT trace (sft_train.jsonl · sft_val.jsonl)

models/                체크포인트 (gitignored)
  step_seed{42,123,7}_*.pt        PART 1 step-wise RL
  sparse_seed{42,123,7}_*.pt      PART 1 sparse RL (ablation)
  sol_a/sft/best/                 PART 2 SFT 어댑터
  sol_a/rl_s{42,123,7}_v2/best/   PART 2 GRPO RL 어댑터 (공격적 설정, 본 실험)

model_releases/
  rag_rl_checkpoints.zip          배포용 통합 zip (~40 MB, 33 파일)

results/                          Table 1, 학습 곡선, 평가 JSON, rollout 덤프
tests/                            단위 테스트 24개 (env, reward)
```

---

## 데이터 출처 및 라이선스

- **HotpotQA distractor set**: CC BY-SA 4.0
- **스포츠 룰북 350문항 (자체 구축)**: 배구, 탁구, 배드민턴, 미식축구, 축구, 농구, 야구, 하키 공식 룰 텍스트를 HotpotQA 포맷으로 직접 라벨링 (OOD 전이 평가 전용)

---

## 결과 시각화

- PART 1 학습 곡선: [results/learning_curves.png](results/learning_curves.png)
- PART 2 학습 곡선: [results/sol_a_learning_curves.png](results/sol_a_learning_curves.png)
- CartPole sanity 곡선: [results/cartpole_curve.png](results/cartpole_curve.png)
- PART 1 결과 표: [results/table1_3seed.json](results/table1_3seed.json) · [results/baseline_table.csv](results/baseline_table.csv)
- PART 2 결과 집계: [results/sol_a_summary.json](results/sol_a_summary.json)
- 정성 사례 분석: [results/sol_a_qualitative_cases.md](results/sol_a_qualitative_cases.md)
