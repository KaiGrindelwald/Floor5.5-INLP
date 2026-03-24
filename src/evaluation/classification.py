"""
src/evaluation/classification.py
----------------------------------
Extended classification metrics: per-class F1, confusion matrix, and a
full classification report saved as JSON. Used by probe_transfer.py (and can
be called standalone) to produce richer per-layer eval output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils.labeling import LABELS_6


def full_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> Dict:
    """Compute extended classification metrics for a fixed label inventory.

    We always evaluate against the full 6-way label space so that:
      - per-class F1 is stable across layers / language splits
      - confusion matrices have consistent shape
      - classification_report does not crash when a split happens to contain
        only a subset of the labels
    """
    if label_names is None:
        label_names = LABELS_6

    labels = list(range(len(label_names)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_cls = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1: Dict[str, float] = {
        lab: float(per_cls[i]) for i, lab in enumerate(label_names)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0,
    )

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def aggregate_by_layer(metrics_by_layer: List[Dict]) -> Dict:
    """Return the best layer for accuracy and macro-F1."""
    accs = [m["acc"] for m in metrics_by_layer]
    f1s = [m["macro_f1"] for m in metrics_by_layer]
    return {
        "best_acc_layer": int(np.argmax(accs)),
        "best_acc": float(max(accs)),
        "best_f1_layer": int(np.argmax(f1s)),
        "best_f1": float(max(f1s)),
    }
