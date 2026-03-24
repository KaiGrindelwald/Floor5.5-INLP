from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from src.utils.io import read_json, read_jsonl, write_json


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_reps(h5_path: str, split: str, lang: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f[split][lang]["reps"][...]


def _load_split_labels(data_jsonl: str, splits_json: str) -> Dict[str, np.ndarray]:
    rows = list(read_jsonl(data_jsonl))
    id2row = {r["id"]: r for r in rows}
    splits = read_json(splits_json)
    return {
        split: np.array([int(id2row[_id]["label_id"]) for _id in ids], dtype=np.int64)
        for split, ids in splits.items()
    }


@dataclass
class ProbeReadout:
    readout_layer: int
    train_lang: str
    pool: str
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    dev_metrics: Optional[Dict] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = np.where(self.scaler_scale == 0.0, 1.0, self.scaler_scale)
        return (X - self.scaler_mean) / denom

    def logits(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform(X.astype(np.float32, copy=False))
        return Z @ self.coef.T + self.intercept[None, :]

    def probs(self, X: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(X))

    def to_dict(self) -> Dict:
        return {
            "readout_layer": int(self.readout_layer),
            "train_lang": self.train_lang,
            "pool": self.pool,
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "classes": self.classes.tolist(),
            "dev_metrics": self.dev_metrics or {},
        }

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ProbeReadout":
        obj = read_json(path)
        return cls(
            readout_layer=int(obj["readout_layer"]),
            train_lang=str(obj["train_lang"]),
            pool=str(obj.get("pool", "cls")),
            scaler_mean=np.array(obj["scaler_mean"], dtype=np.float32),
            scaler_scale=np.array(obj["scaler_scale"], dtype=np.float32),
            coef=np.array(obj["coef"], dtype=np.float32),
            intercept=np.array(obj["intercept"], dtype=np.float32),
            classes=np.array(obj["classes"], dtype=np.int64),
            dev_metrics=obj.get("dev_metrics", {}),
        )


def _subsample(X: np.ndarray, y: np.ndarray, max_n: Optional[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_n is None or len(y) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=max_n, replace=False)
    return X[idx], y[idx]


def _fit_one_probe(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train.astype(np.float32, copy=False))
    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(Xs, y_train)
    return scaler, clf


def fit_probe_readout(
    reps_h5: str,
    data_jsonl: str,
    splits_json: str,
    train_lang: str = "en",
    dev_lang: str = "en",
    readout_layer: Optional[int] = -1,
    pool: str = "cls",
    max_train: Optional[int] = None,
    seed: int = 42,
) -> ProbeReadout:
    """Fit a multinomial linear probe used as the intervention readout.

    readout_layer semantics:
      -1   -> final layer
      None -> select best layer on dev split (macro-F1)
      >=0  -> fixed layer index
    """
    labels = _load_split_labels(data_jsonl, splits_json)
    X_tr = _load_reps(reps_h5, "train", train_lang)
    X_dev = _load_reps(reps_h5, "dev", dev_lang)
    y_tr = labels["train"]
    y_dev = labels["dev"]

    X_tr, y_tr = _subsample(X_tr, y_tr, max_train, seed)

    n_layers = X_tr.shape[1]
    layer_candidates = [n_layers - 1] if readout_layer == -1 else ([int(readout_layer)] if readout_layer is not None else list(range(n_layers)))

    best = None
    best_metrics = None

    for layer in layer_candidates:
        scaler, clf = _fit_one_probe(X_tr[:, layer, :], y_tr)
        dev_logits = scaler.transform(X_dev[:, layer, :].astype(np.float32, copy=False)) @ clf.coef_.T + clf.intercept_[None, :]
        dev_pred = dev_logits.argmax(axis=1)
        metrics = {
            "layer": int(layer),
            "dev_acc": float(accuracy_score(y_dev, dev_pred)),
            "dev_macro_f1": float(f1_score(y_dev, dev_pred, average="macro", zero_division=0)),
        }
        score = (metrics["dev_macro_f1"], metrics["dev_acc"], -layer)
        if best is None or score > best:
            best = score
            best_metrics = (layer, scaler, clf, metrics)

    assert best_metrics is not None
    layer, scaler, clf, metrics = best_metrics
    return ProbeReadout(
        readout_layer=int(layer),
        train_lang=train_lang,
        pool=pool,
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        classes=clf.classes_.astype(np.int64),
        dev_metrics=metrics,
    )
