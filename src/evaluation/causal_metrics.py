"""
src/evaluation/causal_metrics.py
----------------------------------
Causal effect metrics for intervention experiments.

Functions operate on raw logit/probability arrays (NumPy) produced by
the sweep.py layer_sweep and alpha_sweep functions.
"""

from __future__ import annotations

from typing import List

import numpy as np


def delta_logit(
    logits_base: np.ndarray,       # (N, C)
    logits_int:  np.ndarray,       # (N, C)
    target_class: int,
) -> np.ndarray:
    """
    Per-example change in logit for the target class.

    Returns:
        delta: (N,) float array  — positive means the steering increased the logit.
    """
    return (logits_int[:, target_class] - logits_base[:, target_class]).astype(np.float64)


def delta_prob(
    logits_base: np.ndarray,       # (N, C)
    logits_int:  np.ndarray,       # (N, C)
    target_class: int,
) -> np.ndarray:
    """
    Per-example change in softmax probability for the target class.

    Returns:
        delta: (N,) float array.
    """
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    p_base = _softmax(logits_base.astype(np.float64))
    p_int  = _softmax(logits_int.astype(np.float64))
    return p_int[:, target_class] - p_base[:, target_class]


def ate(delta: np.ndarray) -> float:
    """Average Treatment Effect: mean of per-example delta_prob."""
    return float(delta.mean())


def success_rate(
    logits_base: np.ndarray,   # (N, C)
    logits_int:  np.ndarray,   # (N, C)
    target_class: int,
) -> float:
    """
    Fraction of examples where p(target_class) increased after intervention.
    Ideal: 1.0 (intervention always helps).
    """
    dp = delta_prob(logits_base, logits_int, target_class)
    return float((dp > 0).mean())


def summarize(
    logits_base: np.ndarray,
    logits_int:  np.ndarray,
    target_class: int,
) -> dict:
    """Full causal metric summary dict for one layer/alpha."""
    dl = delta_logit(logits_base, logits_int, target_class)
    dp = delta_prob(logits_base,  logits_int, target_class)
    return {
        "delta_logit_mean": float(dl.mean()),
        "delta_logit_std":  float(dl.std()),
        "delta_prob_mean":  float(dp.mean()),
        "delta_prob_std":   float(dp.std()),
        "ate":              ate(dp),
        "success_rate":     success_rate(logits_base, logits_int, target_class),
    }
