"""
src/interventions/evaluate_intervention.py
------------------------------------------
CLI: run layer-sweep or alpha-sweep intervention and save results.

Unlike the earlier version that relied on a randomly initialised classifier
head, this script uses a frozen encoder plus an English-trained linear probe
readout. That makes the resulting Δlogit / Δp curves interpretable and aligned
with the proposal's encoder-centric methodology.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoModel, AutoTokenizer

from src.interventions.readout import ProbeReadout, fit_probe_readout
from src.interventions.sweep import alpha_sweep, layer_sweep
from src.utils.hf import get_device
from src.utils.io import read_json, read_jsonl, write_json
from src.utils.labeling import LABEL2ID, LABELS_6
from src.utils.seed import set_seed


def _load_directions(directions_dir: str, label: str) -> np.ndarray:
    p = Path(directions_dir) / f"{label}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Direction file not found: {p}")
    return np.load(str(p))


def _load_texts(data_jsonl: str, splits_json: str, split: str, lang: str) -> List[str]:
    data = list(read_jsonl(data_jsonl))
    splits = read_json(splits_json)
    id2row = {r["id"]: r for r in data}
    ids = splits[split]
    return [id2row[_id][f"text_{lang}"] for _id in ids]


def _parse_readout_layer(raw: str) -> Optional[int]:
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
    ap = argparse.ArgumentParser(description="Run concept-steering intervention sweep.")
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--reps_h5", type=str, required=True,
                    help="Cached representations used to fit the probe readout.")
    ap.add_argument("--directions_dir", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True,
                    help="HuggingFace encoder model ID (e.g. xlm-roberta-large).")
    ap.add_argument("--lang", type=str, required=True)
    ap.add_argument("--target_label", type=str, required=True, choices=LABELS_6)
    ap.add_argument("--sweep_type", type=str, default="layer", choices=["layer", "alpha"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--best_layer", type=int, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--max_examples", type=int, default=500)
    ap.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    ap.add_argument("--readout_layer", type=str, default="final",
                    help="Probe readout layer: final, auto, or an integer index.")
    ap.add_argument("--max_probe_train", type=int, default=6000)
    ap.add_argument("--readout_json", type=str, default=None,
                    help="Optional cache path for the fitted probe readout.")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.force_cpu)

    print(f"[evaluate_intervention] Loading encoder: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id)
    model.eval()
    model.to(device)

    texts = _load_texts(args.data_jsonl, args.splits_json, args.split, args.lang)
    if args.max_examples and len(texts) > args.max_examples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(texts), size=args.max_examples, replace=False)
        texts = [texts[i] for i in idx]
    print(f"[evaluate_intervention] Examples={len(texts)} lang={args.lang}")

    readout = _load_or_fit_readout(args)
    print(
        f"[evaluate_intervention] Readout layer={readout.readout_layer} "
        f"dev_macro_f1={readout.dev_metrics.get('dev_macro_f1', float('nan')):.4f}"
    )

    target_class = LABEL2ID[args.target_label]
    directions = _load_directions(args.directions_dir, args.target_label)
    print(f"[evaluate_intervention] Directions shape: {directions.shape}")

    out: Dict = {
        "model_id": args.model_id,
        "reps_h5": args.reps_h5,
        "lang": args.lang,
        "target_label": args.target_label,
        "target_class": int(target_class),
        "sweep_type": args.sweep_type,
        "split": args.split,
        "directions_dir": args.directions_dir,
        "readout": readout.to_dict(),
    }

    if args.sweep_type == "layer":
        result = layer_sweep(
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
        )
        out["alpha"] = float(args.alpha)
        out["layer_sweep"] = result
    else:
        best_layer = args.best_layer if args.best_layer is not None else directions.shape[0] - 1
        result = alpha_sweep(
            model=model,
            tokenizer=tok,
            texts=texts,
            device=device,
            direction=directions[best_layer],
            readout=readout,
            layer_idx=best_layer,
            target_class=target_class,
            alphas=args.alphas,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        out["best_layer"] = int(best_layer)
        out["alpha_sweep"] = result

    write_json(args.out_json, out)
    print(f"[evaluate_intervention] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
