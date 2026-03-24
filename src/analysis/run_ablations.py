"""
src/analysis/run_ablations.py
------------------------------
CLI: run ablation controls for a given model/language/concept and save results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

from src.analysis.ablations import random_direction_control, shuffled_concept_control
from src.interventions.readout import ProbeReadout, fit_probe_readout
from src.utils.hf import get_device
from src.utils.io import read_json, read_jsonl, write_json
from src.utils.labeling import LABEL2ID, LABELS_6
from src.utils.seed import set_seed


def _parse_readout_layer(raw: str):
    raw = raw.strip().lower()
    if raw == "final":
        return -1
    if raw == "auto":
        return None
    return int(raw)


def _load_or_fit_readout(args) -> ProbeReadout:
    if args.readout_json and Path(args.readout_json).exists():
        return ProbeReadout.load(args.readout_json)
    readout = fit_probe_readout(
        reps_h5=args.reps_h5,
        data_jsonl=args.data_jsonl,
        splits_json=args.splits_json,
        train_lang="en",
        dev_lang="en",
        readout_layer=_parse_readout_layer(args.readout_layer),
        pool=args.pool,
        max_train=args.max_probe_train,
        seed=args.seed,
    )
    if args.readout_json:
        Path(args.readout_json).parent.mkdir(parents=True, exist_ok=True)
        readout.save(args.readout_json)
    return readout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--reps_h5", type=str, required=True)
    ap.add_argument("--directions_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--lang", type=str, required=True)
    ap.add_argument("--target_label", type=str, required=True, choices=LABELS_6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_examples", type=int, default=300)
    ap.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    ap.add_argument("--readout_layer", type=str, default="final")
    ap.add_argument("--max_probe_train", type=int, default=6000)
    ap.add_argument("--readout_json", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_ablations] Loading encoder: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id)
    model.eval()
    model.to(device)

    data = list(read_jsonl(args.data_jsonl))
    splits = read_json(args.splits_json)
    id2row = {r["id"]: r for r in data}
    ids = splits[args.split]
    texts = [id2row[_id][f"text_{args.lang}"] for _id in ids]
    if args.max_examples and len(texts) > args.max_examples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(texts), size=args.max_examples, replace=False)
        texts = [texts[i] for i in idx]

    readout = _load_or_fit_readout(args)
    directions = np.load(str(Path(args.directions_dir) / f"{args.target_label}.npy"))
    target_class = LABEL2ID[args.target_label]

    r_rnd = random_direction_control(
        model=model,
        tokenizer=tok,
        texts=texts,
        device=device,
        directions=directions,
        readout=readout,
        target_class=target_class,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    r_rnd.update({
        "model_id": args.model_id,
        "reps_h5": args.reps_h5,
        "lang": args.lang,
        "target_label": args.target_label,
        "readout": readout.to_dict(),
    })
    write_json(out_dir / "random_direction.json", r_rnd)

    r_shuf = shuffled_concept_control(
        model=model,
        tokenizer=tok,
        texts=texts,
        device=device,
        directions_dir=args.directions_dir,
        readout=readout,
        target_label=args.target_label,
        surrogate_label=None,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    r_shuf.update({
        "model_id": args.model_id,
        "reps_h5": args.reps_h5,
        "lang": args.lang,
        "target_label": args.target_label,
        "readout": readout.to_dict(),
    })
    write_json(out_dir / "shuffled_concept.json", r_shuf)
    print(f"[run_ablations] Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
