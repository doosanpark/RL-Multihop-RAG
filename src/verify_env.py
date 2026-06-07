"""환경 sanity check: torch GPU + 주요 라이브러리 import 확인.

실행:
    python -m src.verify_env
"""

from __future__ import annotations

import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}\n")

    # torch + CUDA
    import torch

    print(f"torch       : {torch.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"CUDA avail. : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device 0  : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"  mem       : free={free / 1e9:.2f} GB / total={total / 1e9:.2f} GB")

    # 주요 라이브러리
    print()
    libs = [
        "numpy",
        "datasets",
        "transformers",
        "sentence_transformers",
        "faiss",
        "wandb",
        "gymnasium",
        "sklearn",
        "pandas",
        "matplotlib",
    ]
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "?")
            print(f"  {lib:<22}: {ver}")
        except Exception as e:  # noqa: BLE001
            print(f"  {lib:<22}: FAIL ({e.__class__.__name__}: {e})")


if __name__ == "__main__":
    main()
