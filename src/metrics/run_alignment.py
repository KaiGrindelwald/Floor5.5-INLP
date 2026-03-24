from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from src.utils.io import read_json, read_jsonl, write_json
from src.utils.metrics import cosine_similarity_rows, linear_cka
from src.utils.seed import set_seed


def _load_reps(h5_path: str, split: str, lang: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f[split][lang]["reps"][...]


def _cosine_by_layer(A: np.ndarray, B: np.ndarray) -> Dict[str, List[float]]:
    # A,B: (N,L,D)
    assert A.shape == B.shape
    N, L, D = A.shape
    means, stds = [], []
    for l in range(L):
        cos = cosine_similarity_rows(A[:, l, :].astype(np.float32), B[:, l, :].astype(np.float32)).reshape(-1)
        means.append(float(cos.mean()))
        stds.append(float(cos.std()))
    return {"mean": means, "std": stds}


def _cka_by_layer(A: np.ndarray, B: np.ndarray, max_n: int, seed: int) -> List[float]:
    # A,B: (N,L,D)
    assert A.shape == B.shape
    N, L, D = A.shape
    n = min(N, max_n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n, replace=False)
    scores = []
    for l in tqdm(range(L), desc="CKA"):
        X = A[idx, l, :].astype(np.float32)
        Y = B[idx, l, :].astype(np.float32)
        scores.append(linear_cka(X, Y))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--reps_h5", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--cka_max_n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    data = list(read_jsonl(args.data_jsonl))
    splits = read_json(args.splits_json)
    example0 = data[0]
    text_keys = [k for k in example0.keys() if k.startswith("text_")]
    langs = [k.replace("text_", "") for k in sorted(text_keys)]
    # determine LR
    lr_langs = [l for l in langs if l not in ("en", "hi")]
    if len(lr_langs) != 1:
        raise RuntimeError(f"Expected exactly 1 LR language, found {lr_langs}")
    lr = lr_langs[0]

    # We'll compute alignment on test split (midsem focus)
    split = "test"
    A_en = _load_reps(args.reps_h5, split, "en")
    A_hi = _load_reps(args.reps_h5, split, "hi")
    A_lr = _load_reps(args.reps_h5, split, lr)

    out: Dict = {
        "reps_h5": args.reps_h5,
        "split": split,
        "languages": {"en": "en", "hi": "hi", "lr": lr},
        "cosine": {
            f"en_vs_hi": _cosine_by_layer(A_en, A_hi),
            f"en_vs_{lr}": _cosine_by_layer(A_en, A_lr),
        },
        "cka": {
            f"en_vs_hi": _cka_by_layer(A_en, A_hi, max_n=args.cka_max_n, seed=args.seed),
            f"en_vs_{lr}": _cka_by_layer(A_en, A_lr, max_n=args.cka_max_n, seed=args.seed + 1),
        },
    }

    write_json(args.out_json, out)
    print(f"[run_alignment] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
