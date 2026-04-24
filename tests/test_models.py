"""
tests/test_models.py
测试 PLS 和 SVR 模型的 fit/predict 接口。
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import PLSModel, SVRModel, get_model


def _make_data(n=200, f=11, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, f))
    y = X[:, 0] * 2.5 + X[:, 2] - 0.3 * X[:, 5] + rng.normal(0, 0.1, n)
    return X, y


def test_pls_fit_predict():
    X, y = _make_data()
    model = PLSModel(n_components=4)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    # R² 在训练集上应合理
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.5, f"PLS R² 过低: {r2:.3f}"


def test_svr_fit_predict():
    X, y = _make_data()
    model = SVRModel(C=5.0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)


def test_get_model_pls():
    m = get_model("pls", n_components=3)
    assert isinstance(m, PLSModel)


def test_get_model_svr():
    m = get_model("svr", C=1.0)
    assert isinstance(m, SVRModel)


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("unknown_model")


def test_pls_save_load(tmp_path):
    X, y = _make_data()
    model = PLSModel(n_components=3)
    model.fit(X, y)
    path = model.save(tmp_path / "pls.pkl")
    loaded = PLSModel.load(path)
    np.testing.assert_allclose(model.predict(X), loaded.predict(X))


def test_svr_save_load(tmp_path):
    X, y = _make_data()
    model = SVRModel()
    model.fit(X, y)
    path = model.save(tmp_path / "svr.pkl")
    loaded = SVRModel.load(path)
    np.testing.assert_allclose(model.predict(X), loaded.predict(X))
