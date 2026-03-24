import numpy as np

from src.interventions.readout import ProbeReadout


def test_probe_readout_logits_shape():
    readout = ProbeReadout(
        readout_layer=3,
        train_lang="en",
        pool="cls",
        scaler_mean=np.zeros(4, dtype=np.float32),
        scaler_scale=np.ones(4, dtype=np.float32),
        coef=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        intercept=np.array([0.0, 0.5], dtype=np.float32),
        classes=np.array([0, 1], dtype=np.int64),
        dev_metrics={"dev_macro_f1": 1.0},
    )
    X = np.array([[2, 3, 0, 0], [4, 5, 0, 0]], dtype=np.float32)
    logits = readout.logits(X)
    assert logits.shape == (2, 2)
    assert np.allclose(logits[0], np.array([2.0, 3.5], dtype=np.float32))


def test_probe_readout_roundtrip(tmp_path):
    readout = ProbeReadout(
        readout_layer=5,
        train_lang="en",
        pool="mean",
        scaler_mean=np.array([1.0, 2.0], dtype=np.float32),
        scaler_scale=np.array([3.0, 4.0], dtype=np.float32),
        coef=np.array([[0.1, 0.2]], dtype=np.float32),
        intercept=np.array([0.3], dtype=np.float32),
        classes=np.array([0], dtype=np.int64),
        dev_metrics={"dev_acc": 0.5},
    )
    path = tmp_path / "readout.json"
    readout.save(path)
    loaded = ProbeReadout.load(path)
    assert loaded.readout_layer == 5
    assert loaded.pool == "mean"
    assert np.allclose(loaded.scaler_mean, readout.scaler_mean)
    assert np.allclose(loaded.coef, readout.coef)
