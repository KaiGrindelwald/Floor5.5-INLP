from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.utils.hf import get_device, load_encoder
from src.utils.io import read_json, read_jsonl
from src.utils.pooling import pool_hidden
from src.utils.seed import set_seed


def _dtype_from_str(s: str) -> np.dtype:
    s = s.lower()
    if s == "float16":
        return np.float16
    if s == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype={s} (use float16 or float32)")


@torch.no_grad()
def encode_batch(
    bundle,
    texts: List[str],
    pool: str,
    max_length: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Return hidden_states list (len=L+1) each (B,T,D), plus attention_mask."""
    tok = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    attention_mask = enc["attention_mask"].to(device)
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc, output_hidden_states=True, return_dict=True)
    hidden_states = list(out.hidden_states)  # (L+1) * (B,T,D)
    return hidden_states, attention_mask


def write_reps_h5(
    out_h5: Path,
    model_id: str,
    reps_by_split_lang_layer: Dict[str, Dict[str, np.ndarray]],
    meta: Dict,
) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.attrs["model_id"] = model_id
        for k, v in meta.items():
            f.attrs[k] = v

        for split, lang_map in reps_by_split_lang_layer.items():
            g_split = f.create_group(split)
            for lang, arr in lang_map.items():
                # arr: (N, L, D)
                g_lang = g_split.create_group(lang)
                g_lang.create_dataset(
                    "reps",
                    data=arr,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                    chunks=(min(256, arr.shape[0]), 1, arr.shape[2]),
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--out_h5", type=str, required=True)
    ap.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--include_embedding_layer", action="store_true", help="If set, keep hidden_states[0] (embeddings). Otherwise drop it.")
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    data = list(read_jsonl(args.data_jsonl))
    splits = read_json(args.splits_json)
    id2row = {r["id"]: r for r in data}

    # Infer languages from keys
    # We assume text_en and text_hi exist, and exactly one text_<lr> key beyond those.
    example0 = data[0]
    text_keys = [k for k in example0.keys() if k.startswith("text_")]
    # e.g., ["text_en","text_hi","text_cy"]
    langs = [k.replace("text_", "") for k in sorted(text_keys)]
    if "en" not in langs or "hi" not in langs:
        raise RuntimeError(f"Expected text_en and text_hi, found keys={text_keys}")
    print(f"[cache_reps] Languages detected: {langs}")

    device = get_device(force_cpu=args.force_cpu)
    bundle = load_encoder(args.model_id, device)
    np_dtype = _dtype_from_str(args.dtype)

    reps_out: Dict[str, Dict[str, np.ndarray]] = {}
    n_layers_keep = None
    dim = None

    for split in ["train", "dev", "test"]:
        ids = splits[split]
        reps_out[split] = {}

        for lang in langs:
            texts = [id2row[_id][f"text_{lang}"] for _id in ids]
            all_pooled = []

            for i in tqdm(range(0, len(texts), args.batch_size), desc=f"{split}/{lang}"):
                batch = texts[i:i+args.batch_size]
                hidden_states, attention_mask = encode_batch(bundle, batch, pool=args.pool, max_length=args.max_length)
                if not args.include_embedding_layer:
                    hidden_states = hidden_states[1:]

                # pool each layer to (B,D)
                pooled_layers = []
                for h in hidden_states:
                    pooled_layers.append(pool_hidden(h, attention_mask, pool=args.pool).detach().cpu())
                # stack to (B, L, D)
                stacked = torch.stack(pooled_layers, dim=1)
                all_pooled.append(stacked)

                if n_layers_keep is None:
                    n_layers_keep = stacked.shape[1]
                    dim = stacked.shape[2]

            reps = torch.cat(all_pooled, dim=0).numpy().astype(np_dtype, copy=False)
            # reps: (N, L, D)
            reps_out[split][lang] = reps

    meta = {
        "pool": args.pool,
        "max_length": int(args.max_length),
        "dtype": str(np_dtype),
        "n_layers": int(n_layers_keep),
        "dim": int(dim),
        "include_embedding_layer": bool(args.include_embedding_layer),
        "seed": int(args.seed),
    }

    out_h5 = Path(args.out_h5)
    write_reps_h5(out_h5, args.model_id, reps_out, meta)
    print(f"[cache_reps] Wrote {out_h5}")


if __name__ == "__main__":
    main()
