from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.utils.io import read_jsonl
from src.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--n", type=int, default=200, help="How many rows to export for manual verification.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rows = list(read_jsonl(args.data_jsonl))
    n = min(args.n, len(rows))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=n, replace=False)
    picked = [rows[i] for i in idx]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # detect LR key
    k_lr = [k for k in picked[0].keys() if k.startswith("text_") and k not in ("text_en", "text_hi")]
    if len(k_lr) != 1:
        raise RuntimeError(f"Expected exactly one LR text key, found: {k_lr}")
    k_lr = k_lr[0]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "text_en", "text_hi", k_lr])
        for r in picked:
            w.writerow([r["id"], r["label"], r["text_en"], r["text_hi"], r[k_lr]])

    print(f"[export_gold_for_manual] Wrote {out_csv}")


if __name__ == "__main__":
    main()
