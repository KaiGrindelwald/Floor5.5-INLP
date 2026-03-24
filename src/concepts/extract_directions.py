from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils.io import read_json, read_jsonl, write_json
from src.utils.labeling import ID2LABEL, LABEL2ID, LABELS_6
from src.utils.seed import set_seed


def json_dump(obj) -> str:
    """Serialize obj to a pretty-printed JSON string."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _load_reps(h5_path: str, split: str, lang: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f[split][lang]["reps"][...]


def _get_labels_map(data_jsonl: str, splits_json: str) -> Dict[str, Dict[str, np.ndarray]]:
    data = list(read_jsonl(data_jsonl))
    id2 = {r["id"]: r for r in data}
    splits = read_json(splits_json)

    out = {}
    for split, ids in splits.items():
        y = np.array([int(id2[_id]["label_id"]) for _id in ids], dtype=np.int64)
        out[split] = {"ids": np.array(ids), "y": y}
    return out


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--reps_h5", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--method", type=str, nargs="+", default=["mean_diff"],
                    choices=["mean_diff", "probe_weight", "pca_residual"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negatives", type=str, default="rest", choices=["rest", "non_moral"], help="Neg set for mean_diff: all other labels, or only NM.")
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = _get_labels_map(args.data_jsonl, args.splits_json)
    X_tr_en = _load_reps(args.reps_h5, "train", "en")  # (N,L,D)
    y_tr = labels["train"]["y"]

    N, L, D = X_tr_en.shape

    meta = {
        "reps_h5": args.reps_h5,
        "methods": args.method,
        "negatives": args.negatives,
        "seed": int(args.seed),
        "n_layers": int(L),
        "dim": int(D),
        "labels": LABELS_6,
    }
    (out_dir / "meta.json").write_text(json_dump(meta), encoding="utf-8")

    # We'll write: directions/{method}/{label}.npy where array shape (L,D)
    for method in args.method:
        (out_dir / method).mkdir(parents=True, exist_ok=True)

    if "mean_diff" in args.method:
        dirs = {lab: np.zeros((L, D), dtype=np.float32) for lab in LABELS_6}
        for l in tqdm(range(L), desc="mean_diff layers"):
            X = X_tr_en[:, l, :].astype(np.float32)
            for lab in LABELS_6:
                y_pos = LABEL2ID[lab]
                pos = X[y_tr == y_pos]
                if args.negatives == "non_moral":
                    neg = X[y_tr == LABEL2ID["NM"]]
                else:
                    neg = X[y_tr != y_pos]
                mu_pos = pos.mean(axis=0)
                mu_neg = neg.mean(axis=0)
                v = _unit(mu_pos - mu_neg)
                dirs[lab][l, :] = v

        for lab in LABELS_6:
            np.save(out_dir / "mean_diff" / f"{lab}.npy", dirs[lab])

    if "probe_weight" in args.method:
        # One-vs-rest logistic regression; take weight vector as direction
        dirs = {lab: np.zeros((L, D), dtype=np.float32) for lab in LABELS_6}
        for l in tqdm(range(L), desc="probe_weight layers"):
            X = X_tr_en[:, l, :].astype(np.float32)
            for lab in LABELS_6:
                y_pos = LABEL2ID[lab]
                y_bin = (y_tr == y_pos).astype(np.int32)

                clf = Pipeline([
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("lr", LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        class_weight="balanced",
                    )),
                ])
                clf.fit(X, y_bin)
                w = clf.named_steps["lr"].coef_.reshape(-1)  # (D,)
                dirs[lab][l, :] = _unit(w.astype(np.float32))

        for lab in LABELS_6:
            np.save(out_dir / "probe_weight" / f"{lab}.npy", dirs[lab])

    if "pca_residual" in args.method:
        # PCA on the residual between positives and negatives per label.
        # Stack pos+neg horizontally, centre, compute PCA, take PC1 as direction.
        (out_dir / "pca_residual").mkdir(parents=True, exist_ok=True)
        dirs = {lab: np.zeros((L, D), dtype=np.float32) for lab in LABELS_6}
        for l in tqdm(range(L), desc="pca_residual layers"):
            X = X_tr_en[:, l, :].astype(np.float32)
            for lab in LABELS_6:
                y_pos = LABEL2ID[lab]
                pos = X[y_tr == y_pos]
                if args.negatives == "non_moral":
                    neg = X[y_tr == LABEL2ID["NM"]]
                else:
                    neg = X[y_tr != y_pos]
                # Centre each group, then stack into residual matrix
                pos_c = pos - pos.mean(axis=0)
                neg_c = neg - neg.mean(axis=0)
                residual = np.concatenate([pos_c, neg_c], axis=0)
                pca = PCA(n_components=1)
                pca.fit(residual)
                # Ensure direction points from neg to pos (flip if needed)
                v = pca.components_[0].astype(np.float32)  # (D,)
                mu_pos = pos.mean(axis=0)
                mu_neg = neg.mean(axis=0)
                if np.dot(v, mu_pos - mu_neg) < 0:
                    v = -v
                dirs[lab][l, :] = _unit(v)

        for lab in LABELS_6:
            np.save(out_dir / "pca_residual" / f"{lab}.npy", dirs[lab])

    print(f"[extract_directions] Wrote directions to {out_dir}")


if __name__ == "__main__":
    main()
