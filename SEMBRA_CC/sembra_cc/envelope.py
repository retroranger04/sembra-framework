"""Gaussian-process envelope models for p_crit and lambda_limit over (alpha, alpha_p)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sembra_cc import TRAINED_MODELS_DIR

P_CRIT_PATH = TRAINED_MODELS_DIR / "pcrit_envelope.pkl"
LAMBDA_LIMIT_PATH = TRAINED_MODELS_DIR / "lambda_limit_envelope.pkl"


def _make_kernel():
    return 1.0 * RBF(length_scale=[0.3, 0.3]) + WhiteKernel(
        noise_level=1e-6, noise_level_bounds="fixed"
    )


def train_envelope_models(
    limit_point_df: pd.DataFrame, seed: int = 42
) -> Tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Train two GPs (p_crit and lambda_limit) on the limit-point rows."""
    X = limit_point_df[["alpha", "alpha_p"]].to_numpy(dtype=np.float64)
    y_p = limit_point_df["p"].to_numpy(dtype=np.float64)
    y_l = limit_point_df["lambda_0"].to_numpy(dtype=np.float64)

    gp_p = GaussianProcessRegressor(
        kernel=_make_kernel(),
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=seed,
    )
    gp_l = GaussianProcessRegressor(
        kernel=_make_kernel(),
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=seed,
    )

    gp_p.fit(X, y_p)
    gp_l.fit(X, y_l)
    return gp_p, gp_l


def save_envelope_models(
    gp_p: GaussianProcessRegressor,
    gp_l: GaussianProcessRegressor,
    p_crit_path: Path = P_CRIT_PATH,
    lambda_limit_path: Path = LAMBDA_LIMIT_PATH,
) -> None:
    p_crit_path = Path(p_crit_path)
    lambda_limit_path = Path(lambda_limit_path)
    p_crit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(p_crit_path, "wb") as fh:
        cloudpickle.dump(gp_p, fh)
    with open(lambda_limit_path, "wb") as fh:
        cloudpickle.dump(gp_l, fh)


def load_envelope_models(
    p_crit_path: Path = P_CRIT_PATH, lambda_limit_path: Path = LAMBDA_LIMIT_PATH
) -> Tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    p_crit_path = Path(p_crit_path)
    lambda_limit_path = Path(lambda_limit_path)
    if not p_crit_path.exists():
        raise FileNotFoundError(f"{p_crit_path} not found")
    if not lambda_limit_path.exists():
        raise FileNotFoundError(f"{lambda_limit_path} not found")
    with open(p_crit_path, "rb") as fh:
        gp_p = cloudpickle.load(fh)
    with open(lambda_limit_path, "rb") as fh:
        gp_l = cloudpickle.load(fh)
    return gp_p, gp_l


def _predict_envelope(model, alpha, alpha_p):
    a = np.asarray(alpha, dtype=np.float64)
    ap = np.asarray(alpha_p, dtype=np.float64)
    broadcast_shape = np.broadcast_shapes(a.shape, ap.shape)
    all_scalar = broadcast_shape == ()
    a_b = np.broadcast_to(a, broadcast_shape).reshape(-1)
    ap_b = np.broadcast_to(ap, broadcast_shape).reshape(-1)
    X = np.stack([a_b, ap_b], axis=1)
    y = model.predict(X)
    y = y.reshape(broadcast_shape)
    if all_scalar:
        return float(y)
    return y


def predict_p_crit(model, alpha, alpha_p):
    return _predict_envelope(model, alpha, alpha_p)


def predict_lambda_limit(model, alpha, alpha_p):
    return _predict_envelope(model, alpha, alpha_p)
