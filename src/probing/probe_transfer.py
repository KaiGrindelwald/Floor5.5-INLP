from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.evaluation.classification import full_eval, aggregate_by_layer
from src.utils.io import read_json, read_jsonl, write_json
from src.utils.labeling import ID2LABEL, LABELS_6
from src.utils.seed import set_seed


def _load_reps(h5_path: str, split: str, lang: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f[split][lang]["reps"][...]


def _get_labels(data_jsonl: str, splits_json: str) -> Dict[str, Dict[str, np.ndarray]]:
    data = list(read_jsonl(data_jsonl))
    id2 = {r["id"]: r for r in data}
    splits = read_json(splits_json)

    out = {}
    for split, ids in splits.items():
        y = np.array([int(id2[_id]["label_id"]) for _id in ids], dtype=np.int64)
        out[split] = {"ids": np.array(ids), "y": y}
    return out


def _fit_eval_one_layer(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=1,
            class_weight="balanced",
        )),
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return full_eval(y_test, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--reps_h5", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--max_train", type=int, default=8000, help="Cap EN train examples for speed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    labels = _get_labels(args.data_jsonl, args.splits_json)
    example0 = next(read_jsonl(args.data_jsonl))
    langs = [k.replace("text_", "") for k in sorted([k for k in example0.keys() if k.startswith("text_")])]
    lr_langs = [l for l in langs if l not in ("en", "hi")]
    if len(lr_langs) != 1:
        raise RuntimeError(f"Expected exactly 1 LR language, found {lr_langs}")
    lr = lr_langs[0]

    # Load reps
    X_tr_en = _load_reps(args.reps_h5, "train", "en")  # (N,L,D)
    y_tr = labels["train"]["y"]
    if len(y_tr) > args.max_train:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y_tr), size=args.max_train, replace=False)
        X_tr_en = X_tr_en[idx]
        y_tr = y_tr[idx]

    X_te_en = _load_reps(args.reps_h5, "test", "en")
    X_te_hi = _load_reps(args.reps_h5, "test", "hi")
    X_te_lr = _load_reps(args.reps_h5, "test", lr)
    y_te = labels["test"]["y"]

    N, L, D = X_tr_en.shape
    out = {
        "reps_h5": args.reps_h5,
        "train_lang": "en",
        "test_langs": ["en", "hi", lr],
        "n_layers": int(L),
        "metrics_by_layer": {
            "en": [],
            "hi": [],
            lr: [],
        }
    }

    for l in tqdm(range(L), desc="Probe layers"):
        Xtr = X_tr_en[:, l, :].astype(np.float32)
        ytr = y_tr
        out["metrics_by_layer"]["en"].append(_fit_eval_one_layer(Xtr, ytr, X_te_en[:, l, :].astype(np.float32), y_te))
        out["metrics_by_layer"]["hi"].append(_fit_eval_one_layer(Xtr, ytr, X_te_hi[:, l, :].astype(np.float32), y_te))
        out["metrics_by_layer"][lr].append(_fit_eval_one_layer(Xtr, ytr, X_te_lr[:, l, :].astype(np.float32), y_te))

    write_json(args.out_json, out)

    # ── Print aggregate best-layer stats ──────────────────────────────────────
    for lang in ["en", "hi", lr]:
        agg = aggregate_by_layer(out["metrics_by_layer"][lang])
        print(f"  [{lang}] best_acc={agg['best_acc']:.3f} @ layer {agg['best_acc_layer']} "
              f"| best_F1={agg['best_f1']:.3f} @ layer {agg['best_f1_layer']}")

    print(f"[probe_transfer] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
