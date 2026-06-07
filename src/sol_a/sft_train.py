"""LoRA SFT warmup: build_sft_data가 만든 trace로 Qwen LoRA를 지도학습.

생성 토큰(think/search/answer)만 loss, 주입된 <information>·프롬프트는 마스킹(-100).

실행 (smoke):
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.sft_train --limit 200 --epochs 1 --out models/sol_a/sft_smoke
본 학습:
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.sft_train --epochs 3 --out models/sol_a/sft
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .format_utils import build_prompt_messages

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


class TraceDataset(Dataset):
    """레코드 -> (input_ids, labels). 프롬프트/CTX 세그먼트는 labels=-100."""

    def __init__(self, records: List[Dict], tok, max_len: int):
        self.records = records
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        rec = self.records[idx]
        tok = self.tok
        # 프롬프트(시스템+질문) -> assistant 생성 시작까지
        prompt_text = tok.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=True
        )
        ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(ids)

        for seg in rec["segments"]:
            seg_ids = tok(seg["text"], add_special_tokens=False)["input_ids"]
            ids += seg_ids
            labels += (seg_ids if seg["t"] else [-100] * len(seg_ids))

        # 종료 토큰 (생성 멈춤 학습)
        ids.append(tok.eos_token_id)
        labels.append(tok.eos_token_id)

        ids = ids[: self.max_len]
        labels = labels[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def make_collate(pad_id: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    return collate


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        tot += out.loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def sample_generations(model, tok, records, device, k: int = 2) -> None:
    model.eval()
    for rec in records[:k]:
        prompt = tok.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=True
        )
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inp, max_new_tokens=200, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n  Q: {rec['question']}  (gold A: {rec['answer']})")
        print("  GEN:", gen.replace("\n", " ")[:500])
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/sol_a/sft_train.jsonl")
    ap.add_argument("--val", default="data/sol_a/sft_val.jsonl")
    ap.add_argument("--out", default="models/sol_a/sft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--grad-ckpt", type=int, default=1, help="1이면 gradient checkpointing")
    ap.add_argument("--limit", type=int, default=0, help=">0이면 train을 잘라 smoke")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    train_recs = load_jsonl(ROOT / args.train)
    val_recs = load_jsonl(ROOT / args.val)
    if args.limit > 0:
        train_recs = train_recs[: args.limit]
        val_recs = val_recs[: min(len(val_recs), 64)]
    print(f"[data] train={len(train_recs)} val={len(val_recs)}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    if args.grad_ckpt:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    collate = make_collate(tok.pad_token_id)
    train_loader = DataLoader(
        TraceDataset(train_recs, tok, args.max_len), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        TraceDataset(val_recs, tok, args.max_len), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate,
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10, final_div_factor=10,
    )
    print(f"[train] optim steps={total_steps}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    gstep = 0
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                gstep += 1
                if gstep % 20 == 0:
                    avg = running / (20 * args.grad_accum)
                    print(f"  e{epoch} step{gstep} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e}")
                    running = 0.0
                if gstep % args.eval_every == 0:
                    vl = eval_loss(model, val_loader, device)
                    print(f"  [val] step{gstep} val_loss={vl:.4f} (best {best_val:.4f})")
                    if vl < best_val:
                        best_val = vl
                        model.save_pretrained(str(out_dir / "best"))
                        print(f"    -> saved best to {out_dir/'best'}")
        # epoch 끝 val
        vl = eval_loss(model, val_loader, device)
        print(f"[epoch {epoch}] val_loss={vl:.4f}")
        if vl < best_val:
            best_val = vl
            model.save_pretrained(str(out_dir / "best"))
            print(f"  -> saved best to {out_dir/'best'}")

    model.save_pretrained(str(out_dir / "last"))
    tok.save_pretrained(str(out_dir / "last"))
    tok.save_pretrained(str(out_dir / "best"))
    print(f"[done] best_val={best_val:.4f}  adapters in {out_dir}")
    print("\n[generation sanity check on val]")
    sample_generations(model, tok, val_recs, device, k=3)


if __name__ == "__main__":
    main()
