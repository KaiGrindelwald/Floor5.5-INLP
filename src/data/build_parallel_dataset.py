from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from tqdm import tqdm

from src.utils.hf import get_device, load_translator, translate_batch
from src.utils.io import write_json, write_jsonl
from src.utils.labeling import LABELS_6, extract_label_from_row
from src.utils.seed import set_seed


def stratified_sample_indices(label_ids: List[int], n_per_class: int, seed: int) -> List[int]:
    """Return indices for a balanced sample over 6 classes."""
    import random

    set_seed(seed)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(label_ids):
        buckets[y].append(i)

    # shuffle each bucket
    for y in buckets:
        random.shuffle(buckets[y])

    picked: List[int] = []
    for y in range(len(LABELS_6)):
        if y not in buckets or len(buckets[y]) == 0:
            raise RuntimeError(f"No examples for class {y} ({LABELS_6[y]}) in filtered data.")
        take = min(n_per_class, len(buckets[y]))
        picked.extend(buckets[y][:take])

    random.shuffle(picked)
    return picked


def make_splits(ids: List[str], labels: List[int], seed: int, train_frac=0.8, dev_frac=0.1):
    """Stratified split by class label."""
    import random

    set_seed(seed)
    # group by label
    by_lab: Dict[int, List[str]] = defaultdict(list)
    for _id, y in zip(ids, labels):
        by_lab[y].append(_id)

    splits = {"train": [], "dev": [], "test": []}
    for y, lst in by_lab.items():
        random.shuffle(lst)
        n = len(lst)
        n_train = int(round(train_frac * n))
        n_dev = int(round(dev_frac * n))
        n_test = n - n_train - n_dev
        splits["train"].extend(lst[:n_train])
        splits["dev"].extend(lst[n_train:n_train+n_dev])
        splits["test"].extend(lst[n_train+n_dev:])
        assert len(splits["test"]) >= 0

    # shuffle within split for convenience
    for k in splits:
        random.shuffle(splits[k])
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_per_class", type=int, default=500, help="How many EN examples per class (6 classes). Total = 6*n_per_class.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_lang", type=str, default="cy", help="Lower-resource language code. Default: cy (Welsh).")

    ap.add_argument("--dataset_name", type=str, default="MLNTeam-Unical/MoralTextManipulation")
    ap.add_argument("--dataset_config", type=str, default="unconditioned")
    ap.add_argument("--dataset_split", type=str, default="revise")

    ap.add_argument("--translator_en_hi", type=str, default="Helsinki-NLP/opus-mt-en-hi")
    ap.add_argument("--translator_en_lr", type=str, default="Helsinki-NLP/opus-mt-en-cy")
    ap.add_argument("--translate_batch", type=int, default=16)
    ap.add_argument("--translate_max_len", type=int, default=256)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--max_rows_scan", type=int, default=200000, help="Scan at most this many rows before sampling (speed/memory).")
    ap.add_argument("--export_only", action="store_true", help="If set, skips translation and only exports the English dataset to out_dir/english_only.jsonl")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    print(f"[build_parallel_dataset] Loading dataset={args.dataset_name} config={args.dataset_config} split={args.dataset_split}")
    ds = load_dataset(args.dataset_name, name=args.dataset_config, split=args.dataset_split)

    # Filter to 'clean' single-label + non-moral examples
    rows: List[Dict[str, Any]] = []
    label_ids: List[int] = []

    scan_n = min(len(ds), args.max_rows_scan)
    print(f"[build_parallel_dataset] Scanning first {scan_n} rows to find clean examples...")
    for i in tqdm(range(scan_n)):
        r = ds[i]
        try:
            y_id, y_name = extract_label_from_row(r)
        except ValueError:
            continue
        text = (r.get("text") or "").strip()
        if not text:
            continue
        rows.append(r)
        label_ids.append(y_id)

    print(f"[build_parallel_dataset] Clean candidates found: {len(rows)}")

    # Balanced sample
    picked = stratified_sample_indices(label_ids, n_per_class=args.n_per_class, seed=args.seed)
    picked_rows = [rows[i] for i in picked]
    picked_labels = [label_ids[i] for i in picked]

    texts_en = [r["text"].strip() for r in picked_rows]

    if args.export_only:
        english_out = out_dir / "english_only.jsonl"
        export_data = []
        for idx, (r, y_id, en) in enumerate(zip(picked_rows, picked_labels, texts_en)):
            export_data.append({
                "id": f"ex_{idx:06d}",
                "label_id": int(y_id),
                "label": LABELS_6[y_id],
                "domain": str(r.get("domain", "")),
                "text_en": en
            })
        write_jsonl(english_out, export_data)
        print(f"[build_parallel_dataset] Exported {len(export_data)} English rows to {english_out}")
        print("[build_parallel_dataset] Exiting early due to --export_only flag.")
        return

    # Build translation models
    device = get_device(force_cpu=args.force_cpu)
    print(f"[build_parallel_dataset] Using device={device}")
    tr_hi = load_translator(args.translator_en_hi, device)
    tr_lr = load_translator(args.translator_en_lr, device)

    print(f"[build_parallel_dataset] Translating EN -> HI ({args.translator_en_hi}) ...")
    texts_hi = []
    for i in tqdm(range(0, len(texts_en), args.translate_batch)):
        batch = texts_en[i:i+args.translate_batch]
        texts_hi.extend(translate_batch(tr_hi, batch, max_length=args.translate_max_len, batch_size=len(batch)))

    print(f"[build_parallel_dataset] Translating EN -> {args.lr_lang.upper()} ({args.translator_en_lr}) ...")
    texts_lr = []
    for i in tqdm(range(0, len(texts_en), args.translate_batch)):
        batch = texts_en[i:i+args.translate_batch]
        texts_lr.extend(translate_batch(tr_lr, batch, max_length=args.translate_max_len, batch_size=len(batch)))

    assert len(texts_hi) == len(texts_en) == len(texts_lr)

    # Create stable ids
    examples: List[Dict[str, Any]] = []
    for idx, (r, y_id, en, hi, lr) in enumerate(zip(picked_rows, picked_labels, texts_en, texts_hi, texts_lr)):
        y_name = LABELS_6[y_id]
        ex_id = f"ex_{idx:06d}"
        examples.append({
            "id": ex_id,
            "label_id": int(y_id),
            "label": y_name,
            "labels_onehot": [1 if k == y_id else 0 for k in range(len(LABELS_6))],
            "domain": str(r.get("domain", "")),
            "text_en": en,
            "text_hi": hi,
            f"text_{args.lr_lang}": lr,
        })

    # Splits
    ids = [e["id"] for e in examples]
    splits = make_splits(ids, [e["label_id"] for e in examples], seed=args.seed)
    write_json(out_dir / "splits.json", splits)
    write_jsonl(out_dir / "parallel.jsonl", examples)

    # Small summary
    counts = defaultdict(int)
    for e in examples:
        counts[e["label"]] += 1
    summary = {"total": len(examples), "per_label": dict(sorted(counts.items()))}
    write_json(out_dir / "summary.json", summary)
    print("[build_parallel_dataset] Wrote:")
    print(f"  - {out_dir/'parallel.jsonl'}")
    print(f"  - {out_dir/'splits.json'}")
    print(f"  - {out_dir/'summary.json'}")
    print("[build_parallel_dataset] Label distribution:")
    for k, v in summary["per_label"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
