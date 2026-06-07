"""GRPO RL: SFT LoRA를 멀티턴 search rollout의 answer-F1 보상으로 파인튜닝.

- 질문당 G rollout -> group-normalized advantage (value network 불필요, 8GB 친화).
- action 토큰(모델 생성)만 정책경사. KL(참조=SFT 초기정책) 페널티로 포맷붕괴 방어.
- dev-best 체크포인트(greedy rollout F1).

smoke:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.grpo_train --steps 8 --batch-q 4 --group 4 --dev-n 0 --out models/sol_a/rl_smoke
본 학습:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.grpo_train --steps 200 --batch-q 8 --group 5 --seed 42 --out models/sol_a/rl_s42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rewards import compute_answer_f1
from .format_utils import Retriever
from .hotpot_data import load_examples
from .reward_a import compute_reward
from .search_env import RolloutConfig, Rollout, rollout_once

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _rng_state() -> Dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(s: Dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_ckpt(ckpt_dir: Path, policy, opt, step: int, best_dev: float, args) -> None:
    """이어하기용 완전 체크포인트: 어댑터 + optimizer + step + best_dev + RNG."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(ckpt_dir / "adapter"))
    torch.save(
        {
            "step": step,
            "best_dev": best_dev,
            "optimizer": opt.state_dict(),
            "rng": _rng_state(),
            "args": vars(args),
        },
        ckpt_dir / "trainer_state.pt",
    )


def seq_logps(model, full_ids: List[int], action_mask: List[int], device: str):
    """action 토큰들의 logp 텐서 반환 (shift 정렬). grad 흐름은 model 모드에 따름."""
    ids = torch.tensor([full_ids], device=device)
    logits = model(ids).logits[0]                  # [T, V]
    shift_logits = logits[:-1]                     # 위치 t -> 토큰 t+1 예측
    shift_labels = ids[0, 1:]                       # [T-1]
    nll = F.cross_entropy(shift_logits, shift_labels, reduction="none")  # [T-1]
    logp = -nll
    amask = torch.tensor(action_mask[1:], device=device, dtype=torch.bool)  # 토큰 t+1이 action?
    return logp[amask]                              # [n_action]


@torch.no_grad()
def dev_eval(model, tok, retriever, dev_examples, cfg, device) -> Dict[str, float]:
    model.eval()
    f1s, ems, ns = [], [], []
    g_cfg = RolloutConfig(**{**cfg.__dict__, "do_sample": False})
    for ex in dev_examples:
        r = rollout_once(model, tok, retriever, ex.question, ex.candidates, g_cfg, device)
        rb = compute_reward(r.gen_text, ex.answer)
        f1s.append(rb.f1); ems.append(rb.em); ns.append(r.n_search)
    model.train()
    n = max(len(f1s), 1)
    return {"f1": sum(f1s)/n, "em": sum(ems)/n, "avg_search": sum(ns)/n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/sol_a/sft/best")
    ap.add_argument("--out", default="models/sol_a/rl")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-q", type=int, default=8, help="step당 질문 수")
    ap.add_argument("--group", type=int, default=5, help="질문당 rollout 수 G")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--fmt-bonus", type=float, default=0.1)
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--dev-n", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=25, help="N step마다 이어하기용 ckpt 저장")
    ap.add_argument("--resume", default="", help="ckpt 디렉터리 경로. 주면 그 지점부터 이어서")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 이어하기: ckpt가 있으면 그 어댑터/상태에서 시작, 없으면 SFT 어댑터에서 시작
    resume_state = None
    if args.resume:
        resume_dir = Path(args.resume)
        adapter_src = str(resume_dir / "adapter")
        resume_state = torch.load(resume_dir / "trainer_state.pt", weights_only=False)
        print(f"[resume] {resume_dir}  (step {resume_state['step']}, best_dev {resume_state['best_dev']:.3f})")
    else:
        adapter_src = args.adapter
        print(f"[load] policy (base+SFT LoRA: {adapter_src}, trainable)")

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    policy = PeftModel.from_pretrained(base, adapter_src, is_trainable=True).to(device)
    policy.config.use_cache = True   # rollout 생성 속도; 학습 forward는 짧음
    policy.print_trainable_parameters()

    ref = None
    if args.kl_coef > 0:
        print("[load] reference (frozen SFT)")
        base2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
        ref = PeftModel.from_pretrained(base2, args.adapter).to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    retriever = Retriever(device=device)
    cfg = RolloutConfig(max_turns=args.max_turns, temperature=args.temperature, do_sample=True)

    train_ex = load_examples("train", n=args.train_n, start=0)
    dev_ex = load_examples("validation", n=args.dev_n, start=0) if args.dev_n > 0 else []
    print(f"[data] train={len(train_ex)} dev={len(dev_ex)}")

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out_dir / "best"))
    hist: List[Dict] = []
    best_dev = -1.0
    start_step = 0

    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state["step"])
        best_dev = float(resume_state["best_dev"])
        _set_rng_state(resume_state["rng"])
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
        print(f"[resume] continuing from step {start_step+1}, best_dev={best_dev:.3f}")

    for step in range(start_step + 1, args.steps + 1):
        qs = random.sample(train_ex, min(args.batch_q, len(train_ex)))

        # ---- rollout (no grad) ----
        policy.eval()
        groups: List[List[Rollout]] = []
        rewards: List[List[float]] = []
        metr = {"f1": [], "em": [], "nsearch": [], "has_ans": [], "stop_answer": []}
        for ex in qs:
            rolls, rs = [], []
            for _ in range(args.group):
                r = rollout_once(policy, tok, retriever, ex.question, ex.candidates, cfg, device)
                rb = compute_reward(r.gen_text, ex.answer, fmt_bonus=args.fmt_bonus)
                rolls.append(r); rs.append(rb.total)
                metr["f1"].append(rb.f1); metr["em"].append(rb.em)
                metr["nsearch"].append(r.n_search); metr["has_ans"].append(float(rb.has_answer))
                metr["stop_answer"].append(float(r.stop_reason == "answer"))
            groups.append(rolls); rewards.append(rs)

        # ---- advantage (group-normalized) ----
        advs: List[List[float]] = []
        for rs in rewards:
            arr = np.array(rs, dtype=np.float32)
            mean = arr.mean()
            std = arr.std()
            advs.append(list((arr - mean) / (std + 1e-4)) if std > 1e-6 else [0.0]*len(arr))

        # ---- policy update (micro-batch=1 누적) ----
        policy.train()
        opt.zero_grad()
        pg_tot, kl_tot, n_tok_tot, n_seq = 0.0, 0.0, 0, 0
        for gi, rolls in enumerate(groups):
            for ri, r in enumerate(rolls):
                adv = advs[gi][ri]
                if adv == 0.0:
                    continue
                if sum(r.action_mask) < 1:
                    continue
                logp = seq_logps(policy, r.full_ids, r.action_mask, device)  # [n_act] grad
                if logp.numel() == 0:
                    continue
                pg = -(adv * logp).mean()
                loss = pg
                if ref is not None and args.kl_coef > 0:
                    with torch.no_grad():
                        logp_ref = seq_logps(ref, r.full_ids, r.action_mask, device)
                    # GRPO k3 추정량: exp(rd) - rd - 1, rd = logp_ref - logp
                    rd = (logp_ref - logp)
                    kl = (torch.exp(rd) - rd - 1).mean()
                    loss = pg + args.kl_coef * kl
                    kl_tot += float(kl.detach())
                loss.backward()
                pg_tot += float(pg.detach()); n_tok_tot += int(logp.numel()); n_seq += 1
        if n_seq > 0:
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
            opt.step()
        opt.zero_grad()

        m = {k: float(np.mean(v)) if v else 0.0 for k, v in metr.items()}
        rec = {"step": step, "reward_f1": m["f1"], "em": m["em"], "avg_search": m["nsearch"],
               "has_ans": m["has_ans"], "stop_answer": m["stop_answer"],
               "pg": pg_tot/max(n_seq,1), "kl": kl_tot/max(n_seq,1), "n_upd_seq": n_seq}
        hist.append(rec)
        print(f"step{step:3d} F1={m['f1']:.3f} EM={m['em']:.3f} search={m['nsearch']:.2f} "
              f"hasA={m['has_ans']:.2f} stopA={m['stop_answer']:.2f} "
              f"pg={rec['pg']:+.3f} kl={rec['kl']:.3f} upd={n_seq}")

        if dev_ex and step % args.eval_every == 0:
            d = dev_eval(policy, tok, retriever, dev_ex, cfg, device)
            print(f"  [dev] step{step} F1={d['f1']:.3f} EM={d['em']:.3f} search={d['avg_search']:.2f} (best {best_dev:.3f})")
            rec["dev_f1"] = d["f1"]; rec["dev_em"] = d["em"]
            if d["f1"] > best_dev:
                best_dev = d["f1"]
                policy.save_pretrained(str(out_dir / "best"))
                print(f"    -> saved dev-best ({best_dev:.3f})")
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)

        if args.save_every > 0 and step % args.save_every == 0:
            save_ckpt(out_dir / "ckpt", policy, opt, step, best_dev, args)
            print(f"  [ckpt] saved resume point at step {step} -> {out_dir/'ckpt'}")

    policy.save_pretrained(str(out_dir / "last"))
    print(f"[done] best_dev_f1={best_dev:.3f}  adapters in {out_dir}")


if __name__ == "__main__":
    main()
