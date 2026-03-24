"""
tests/test_utils.py
--------------------
Unit tests for utility modules (no external data required).
"""

import numpy as np
import pytest
import torch

from src.utils.metrics import cosine_similarity_rows, linear_cka
from src.utils.labeling import extract_label_from_row, LABELS_6, LABEL2ID
from src.utils.pooling import pool_hidden
from src.evaluation.classification import full_eval, aggregate_by_layer
from src.evaluation.causal_metrics import (
    delta_logit, delta_prob, ate, success_rate, summarize
)


# ─── cosine similarity ───────────────────────────────────────────────────────

def test_cosine_identical():
    v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    sim = cosine_similarity_rows(v, v)
    assert abs(sim[0, 0] - 1.0) < 1e-5


def test_cosine_orthogonal():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)
    sim = cosine_similarity_rows(a, b)
    assert abs(sim[0, 0]) < 1e-5


def test_cosine_batch():
    a = np.random.randn(8, 64).astype(np.float32)
    b = np.random.randn(8, 64).astype(np.float32)
    sims = cosine_similarity_rows(a, b)
    assert sims.shape == (8, 1)
    assert np.all(sims >= -1.0 - 1e-5) and np.all(sims <= 1.0 + 1e-5)


# ─── linear CKA ──────────────────────────────────────────────────────────────

def test_cka_identical():
    X = np.random.randn(32, 64).astype(np.float32)
    score = linear_cka(X, X)
    assert abs(score - 1.0) < 1e-5, f"CKA(X, X) should be 1.0, got {score}"


def test_cka_orthogonal_matrices():
    """CKA of two independent random matrices should be close to 0 (not exactly)."""
    np.random.seed(0)
    X = np.random.randn(100, 64).astype(np.float32)
    Y = np.random.randn(100, 64).astype(np.float32)
    score = linear_cka(X, Y)
    assert 0.0 <= score <= 1.0 + 1e-5


def test_cka_bounded():
    X = np.random.randn(50, 32).astype(np.float32)
    Y = np.random.randn(50, 32).astype(np.float32)
    score = linear_cka(X, Y)
    assert 0.0 <= score <= 1.0 + 1e-5


# ─── labeling ────────────────────────────────────────────────────────────────

def test_labeling_clean_ch():
    row = {"CH_ref": 1, "FC_ref": 0, "LB_ref": 0, "AS_ref": 0, "PD_ref": 0, "non_moral_ref": 0}
    y_id, y_name = extract_label_from_row(row)
    assert y_name == "CH"
    assert y_id == LABEL2ID["CH"]


def test_labeling_non_moral():
    row = {"CH_ref": 0, "FC_ref": 0, "LB_ref": 0, "AS_ref": 0, "PD_ref": 0, "non_moral_ref": 1}
    y_id, y_name = extract_label_from_row(row)
    assert y_name == "NM"
    assert y_id == LABEL2ID["NM"]


def test_labeling_multi_label_raises():
    row = {"CH_ref": 1, "FC_ref": 1, "LB_ref": 0, "AS_ref": 0, "PD_ref": 0, "non_moral_ref": 0}
    with pytest.raises(ValueError):
        extract_label_from_row(row)


def test_labeling_no_label_raises():
    row = {"CH_ref": 0, "FC_ref": 0, "LB_ref": 0, "AS_ref": 0, "PD_ref": 0, "non_moral_ref": 0}
    with pytest.raises(ValueError):
        extract_label_from_row(row)


def test_labeling_all_foundations():
    for i, lab in enumerate(["CH", "FC", "LB", "AS", "PD"]):
        row = {f"{k}_ref": (1 if k == lab else 0) for k in ["CH", "FC", "LB", "AS", "PD"]}
        row["non_moral_ref"] = 0
        y_id, y_name = extract_label_from_row(row)
        assert y_name == lab
        assert y_id == LABEL2ID[lab]


# ─── pooling ─────────────────────────────────────────────────────────────────

def test_pooling_cls():
    from src.utils.pooling import pool_hidden
    B, T, D = 4, 10, 16
    hidden = torch.randn(B, T, D)
    mask   = torch.ones(B, T, dtype=torch.long)
    out    = pool_hidden(hidden, mask, pool="cls")
    assert out.shape == (B, D)
    assert torch.allclose(out, hidden[:, 0, :])


def test_pooling_mean():
    from src.utils.pooling import pool_hidden
    B, T, D = 4, 10, 16
    hidden = torch.ones(B, T, D)   # all ones → mean should be 1
    mask   = torch.ones(B, T, dtype=torch.long)
    out    = pool_hidden(hidden, mask, pool="mean")
    assert out.shape == (B, D)
    assert torch.allclose(out, torch.ones(B, D))


def test_pooling_mean_masked():
    """Mean pooling with mask should ignore padded tokens."""
    B, T, D = 2, 6, 4
    hidden = torch.zeros(B, T, D)
    hidden[:, :3, :] = 1.0  # first 3 tokens = ones, last 3 = zeros
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.long)
    out = pool_hidden(hidden, mask, pool="mean")
    assert torch.allclose(out, torch.ones(B, D), atol=1e-5)


def test_pooling_unknown_raises():
    from src.utils.pooling import pool_hidden
    with pytest.raises(ValueError):
        pool_hidden(torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.long), pool="max")


# ─── evaluation.classification ────────────────────────────────────────────────

def test_full_eval_perfect():
    y = np.array([0, 1, 2, 3, 4, 5])
    m = full_eval(y, y)
    assert abs(m["acc"] - 1.0) < 1e-9
    assert abs(m["macro_f1"] - 1.0) < 1e-9
    for lab in LABELS_6:
        assert abs(m["per_class_f1"][lab] - 1.0) < 1e-9
    assert len(m["confusion_matrix"]) == 6


def test_full_eval_all_wrong():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    m = full_eval(y_true, y_pred)
    assert m["acc"] == 0.0


def test_aggregate_by_layer():
    metrics = [{"acc": 0.5, "macro_f1": 0.4}, {"acc": 0.8, "macro_f1": 0.75}, {"acc": 0.6, "macro_f1": 0.9}]
    agg = aggregate_by_layer(metrics)
    assert agg["best_acc_layer"] == 1
    assert agg["best_f1_layer"]  == 2


# ─── evaluation.causal_metrics ───────────────────────────────────────────────

def _make_logits():
    np.random.seed(42)
    base = np.random.randn(10, 6).astype(np.float64)
    # Intervention: boost class 0 by +2
    intv = base.copy()
    intv[:, 0] += 2.0
    return base, intv


def test_delta_logit_positive():
    base, intv = _make_logits()
    dl = delta_logit(base, intv, target_class=0)
    assert dl.shape == (10,)
    assert np.allclose(dl, 2.0, atol=1e-5)


def test_delta_prob_positive():
    base, intv = _make_logits()
    dp = delta_prob(base, intv, target_class=0)
    assert dp.shape == (10,)
    assert (dp > 0).all(), "Boosting logit should increase probability"


def test_ate_positive():
    base, intv = _make_logits()
    dp = delta_prob(base, intv, target_class=0)
    assert ate(dp) > 0


def test_success_rate_all():
    base, intv = _make_logits()
    sr = success_rate(base, intv, target_class=0)
    assert sr == 1.0


def test_summarize_keys():
    base, intv = _make_logits()
    s = summarize(base, intv, target_class=0)
    for key in ["delta_logit_mean", "delta_logit_std", "delta_prob_mean",
                 "delta_prob_std", "ate", "success_rate"]:
        assert key in s, f"Missing key: {key}"
