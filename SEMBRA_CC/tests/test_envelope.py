"""Tests for sembra_cc.envelope."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sembra_cc.envelope import (
    load_envelope_models,
    predict_lambda_limit,
    predict_p_crit,
    save_envelope_models,
    train_envelope_models,
)


def _fixture_limit_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    alphas = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
    alpha_ps = np.linspace(0.0, 0.4, 21)
    rows = []
    for a in alphas:
        for ap in alpha_ps:
            rows.append(
                {
                    "alpha": float(a),
                    "alpha_p": float(ap),
                    "p": float(0.5 + 0.1 * a - 0.5 * ap + 0.01 * rng.standard_normal()),
                    "lambda_0": float(1.7 + 0.1 * a - 0.2 * ap),
                    "eta_0": 0.0,
                    "det_H": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_train_produces_two_models():
    df = _fixture_limit_df()
    gp_p, gp_l = train_envelope_models(df)
    assert gp_p is not None and gp_l is not None


def test_predict_at_training_points_accurate():
    df = _fixture_limit_df()
    gp_p, gp_l = train_envelope_models(df)
    y_p = gp_p.predict(df[["alpha", "alpha_p"]].to_numpy())
    rel = np.abs(y_p - df["p"].to_numpy()) / np.abs(df["p"].to_numpy())
    assert rel.max() < 0.05  # smooth fixture, accuracy is loose to allow noise


def test_roundtrip_serialization(tmp_path):
    df = _fixture_limit_df()
    gp_p, gp_l = train_envelope_models(df)
    p_path = tmp_path / "p.pkl"
    l_path = tmp_path / "l.pkl"
    save_envelope_models(gp_p, gp_l, p_path, l_path)
    gp_p2, gp_l2 = load_envelope_models(p_path, l_path)
    X = np.array([[1.5, 0.1], [1.7, 0.2]])
    assert np.allclose(gp_p.predict(X), gp_p2.predict(X))
    assert np.allclose(gp_l.predict(X), gp_l2.predict(X))


def test_predict_helpers_scalar_and_array():
    df = _fixture_limit_df()
    gp_p, gp_l = train_envelope_models(df)
    s = predict_p_crit(gp_p, 1.5, 0.1)
    assert isinstance(s, float)
    arr = predict_p_crit(gp_p, np.array([1.4, 1.6]), np.array([0.1, 0.2]))
    assert isinstance(arr, np.ndarray) and arr.shape == (2,)
    s2 = predict_lambda_limit(gp_l, 1.5, 0.1)
    assert isinstance(s2, float)


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_envelope_models(tmp_path / "nope.pkl", tmp_path / "nope2.pkl")
