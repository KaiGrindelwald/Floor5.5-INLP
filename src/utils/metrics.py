from __future__ import annotations

import numpy as np


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between a and b. a,b: (N,D)"""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    denom = (a_norm * b_norm).clip(min=eps)
    return (a * b).sum(axis=1, keepdims=True) / denom


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """Linear CKA between two representation matrices (N,Dx) and (N,Dy).

    Uses the efficient formulation from Kornblith et al. (2019):
      CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    after centering X and Y over the sample axis.
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XT_Y = X.T @ Y
    num = np.sum(XT_Y * XT_Y)

    XT_X = X.T @ X
    YT_Y = Y.T @ Y
    denom = np.sqrt(np.sum(XT_X * XT_X) * np.sum(YT_Y * YT_Y)) + eps
    return float(num / denom)
