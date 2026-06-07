"""Solution A 평가: search-protocol rollout으로 in-domain(HotpotQA val) / transfer(sports) F1·EM.

adapter='none'면 frozen base Qwen(=cold-start, 포맷 못 따름 예상).
hop 타입별(bridge/comparison) 집계.

실행:
    # SFT 어댑터, in-domain 200
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.eval_a --adapter models/sol_a/sft/best --dataset hotpot --n 200
    # RL 어댑터, sports transfer
    .\.venv\Scripts\python.exe -u -X utf8 -m src.sol_a.eval_a --adapter models/sol_a/rl_s42/best --dataset sports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .format_utils import Retriever
from .hotpot_data import Example, load_examples
from .reward_a import compute_em
from ..rewards import compute_answer_f1
from .search_env import RolloutConfig, rollout_once

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _join(sents: List[str]) -> str:
    return " ".join(s.strip() for s in sents).strip()


def load_sports(path: Path, n: int | None) -> List[Example]:
    data = json.load(path.open(encoding="utf-8"))
    if n:
        data = data[:n]
    out: List[Example] = []
    for s in data:
        ctx = s["context"]  # [[title, [sents]], ...]
        candidates = [(t, _join(sl)) for t, sl in ctx]
        title_to_text = dict(candidates)
        sf = s["supporting_facts"]  # [[title, sid], ...]
        gold_titles: List[str] = []
        for t, _sid in sf:
            if t not in gold_titles:
                gold_titles.append(t)
        out.append(Example(
            id=s.get("_id", s.get("id", "")),
            question=s["question"].strip(),
            answer=s["answer"].strip(),
            qtype=s.get("type", ""),
            level=s.get("level", ""),
            candidates=candidates,
            gold_titles=gold_titles,
            gold_passages=[(t, title_to_text.get(t, "")) for t in gold_titles],
            gold_sentences={},
        ))
    return out


def aggregate(rows: List[Dict]) -> Dict:
    def agg(subset: List[Dict]) -> Dict:
        n = len(subset)
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "f1": sum(r["f1"] for r in subset) / n,
            "em": sum(r["em"] for r in subset) / n,
            "avg_search": sum(r["n_search"] for r in subset) / n,
            "has_ans": sum(r["has_ans"] for r in subset) / n,
        }
    res = {"overall": agg(rows)}
    for qt in sorted({r["qtype"] for r in rows}):
        res[qt] = agg([r for r in rows if r["qtype"] == qt])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/sol_a/sft/best", help="'none'이면 frozen base")
    ap.add_argument("--dataset", choices=["hotpot", "sports"], default="hotpot")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="", help="rollout 텍스트 덤프 jsonl 경로(선택)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    if args.adapter and args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(base, args.adapter).to(device)
        tag = args.adapter
    else:
        model = base
        tag = "frozen-base"
    model.eval()

    retriever = Retriever(device=device)
    cfg = RolloutConfig(max_turns=args.max_turns, do_sample=False)

    if args.dataset == "hotpot":
        examples = load_examples("validation", n=args.n, start=args.start)
    else:
        examples = load_sports(ROOT / "data/eval/sports.json", args.n if args.n > 0 else None)
    print(f"[eval] model={tag} dataset={args.dataset} n={len(examples)}")

    rows: List[Dict] = []
    dump_f = open(ROOT / args.dump, "w", encoding="utf-8") if args.dump else None
    for i, ex in enumerate(examples):
        r = rollout_once(model, tok, retriever, ex.question, ex.candidates, cfg, device)
        pred = r.answer or ""
        row = {
            "id": ex.id, "qtype": ex.qtype, "f1": compute_answer_f1(pred, ex.answer),
            "em": compute_em(pred, ex.answer), "n_search": r.n_search,
            "has_ans": float(r.answer is not None),
        }
        rows.append(row)
        if dump_f:
            dump_f.write(json.dumps({**row, "question": ex.question, "gold": ex.answer,
                                     "pred": pred, "gen": r.gen_text}, ensure_ascii=False) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  ..{i+1}/{len(examples)}")
    if dump_f:
        dump_f.close()

    res = aggregate(rows)
    res["_meta"] = {"model": tag, "dataset": args.dataset, "n": len(examples), "max_turns": args.max_turns}
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out = args.out or f"results/sol_a_eval_{args.dataset}.json"
    out_path = ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
