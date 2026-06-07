"""학습된 어댑터·체크포인트를 단일 zip으로 묶어 model_releases/에 저장.

포함:
- 2막 LoRA 어댑터 4개 (SFT + RL s42/s123/s7) — best/ + history.json
- 1막 step-wise RL 체크포인트 3개 (best.pt for seed 42/123/7)
- 1막 sparse RL 체크포인트 3개 (ablation 비교용)

GitHub raw URL로 직접 다운로드 가능 (push 후).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "model_releases"
OUT_ZIP = OUT_DIR / "rag_rl_checkpoints.zip"


PART2_ADAPTERS = {
    "part2_sft": "models/sol_a/sft/best",
    "part2_rl_s42": "models/sol_a/rl_s42_v2/best",
    "part2_rl_s123": "models/sol_a/rl_s123_v2/best",
    "part2_rl_s7": "models/sol_a/rl_s7_v2/best",
}

PART2_HISTORY = {
    "part2_rl_s42": "models/sol_a/rl_s42_v2/history.json",
    "part2_rl_s123": "models/sol_a/rl_s123_v2/history.json",
    "part2_rl_s7": "models/sol_a/rl_s7_v2/history.json",
}

PART1_CKPTS = {
    "part1_step_seed42_best.pt": "models/step_seed42_best.pt",
    "part1_step_seed123_best.pt": "models/step_seed123_best.pt",
    "part1_step_seed7_best.pt": "models/step_seed7_best.pt",
    "part1_sparse_seed42_best.pt": "models/sparse_seed42_best.pt",
    "part1_sparse_seed123_best.pt": "models/sparse_seed123_best.pt",
    "part1_sparse_seed7_best.pt": "models/sparse_seed7_best.pt",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_files, total_bytes = 0, 0

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 2막 어댑터
        for name, path in PART2_ADAPTERS.items():
            adapter_dir = ROOT / path
            if not adapter_dir.exists():
                print(f"  [skip] missing: {path}")
                continue
            for f in adapter_dir.rglob("*"):
                if f.is_file():
                    arc = f"{name}/{f.relative_to(adapter_dir)}"
                    zf.write(f, arc)
                    total_files += 1
                    total_bytes += f.stat().st_size

        # 2막 history.json
        for name, path in PART2_HISTORY.items():
            h = ROOT / path
            if h.exists():
                zf.write(h, f"{name}_history.json")
                total_files += 1
                total_bytes += h.stat().st_size

        # 1막 체크포인트
        for arc_name, path in PART1_CKPTS.items():
            p = ROOT / path
            if not p.exists():
                print(f"  [skip] missing: {path}")
                continue
            zf.write(p, arc_name)
            total_files += 1
            total_bytes += p.stat().st_size

    zip_size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    raw_mb = total_bytes / (1024 * 1024)
    print(f"\n[saved] {OUT_ZIP}")
    print(f"  파일 수: {total_files}")
    print(f"  원본 합: {raw_mb:.1f} MB")
    print(f"  압축 후: {zip_size_mb:.1f} MB ({zip_size_mb/raw_mb*100:.0f}%)")


if __name__ == "__main__":
    main()
