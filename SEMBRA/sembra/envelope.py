"""Envelope regressors for p_crit and lambda_0 at the limit point.

Two Gaussian-process surrogates over the 2-D parameter plane ``(alpha,
alpha_p)``, trained on the 125 limit-point rows of ``Real_Data.csv``. Each
uses the anisotropic-RBF + WhiteKernel composition prescribed in
PRD Section 6.4 with ``normalize_y=True`` and ``n_restarts_optimizer=10``.
"""

from __future__ import annotations

from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from sembra import TRAINED_MODELS_DIR

P_CRIT_PATH = TRAINED_MODELS_DIR / "pcrit_envelope.pkl"
LAMBDA_LIMIT_PATH = TRAINED_MODELS_DIR / "lambda_limit_envelope.pkl"


def _build_kernel() -> RBF:
    return 1.0 * RBF(length_scale=[0.3, 0.3]) + WhiteKernel(
        noise_level=1e-6, noise_level_bounds="fixed"
    )


def _fit_single(features: np.ndarray, target: np.ndarray, seed: int) -> GaussianProcessRegressor:
    gp = GaussianProcessRegressor(
        kernel=_build_kernel(),
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=seed,
    )
    gp.fit(features, target)
    return gp


def train_envelope_models(
    limit_point_df: pd.DataFrame,
    seed: int = 42,
) -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Train and persist the two envelope GP regressors.

    Parameters
    ----------
    limit_point_df
        The 125 limit-point rows (``det_H < 0``) from the dataset.
    seed
        Random seed forwarded to ``GaussianProcessRegressor`` for restart
        initialization.

    Returns
    -------
    tuple
        ``(p_crit_model, lambda_limit_model)``. Both are also saved to disk
        at ``trained_models/pcrit_envelope.pkl`` and
        ``trained_models/lambda_limit_envelope.pkl``.

    Raises
    ------
    RuntimeError
        If GP fitting raises any exception (PRD Section 6.4).
    """
    if limit_point_df.empty:
        raise ValueError("limit_point_df is empty; expected 125 limit-point rows.")
    features = limit_point_df[["alpha", "alpha_p"]].to_numpy(dtype=float)
    p_targets = limit_point_df["p"].to_numpy(dtype=float)
    lambda_targets = limit_point_df["lambda_0"].to_numpy(dtype=float)

    try:
        p_crit_model = _fit_single(features, p_targets, seed)
        lambda_limit_model = _fit_single(features, lambda_targets, seed)
    except Exception as exc:
        raise RuntimeError(f"Gaussian-process envelope training failed: {exc}") from exc

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(P_CRIT_PATH, "wb") as fh:
        cloudpickle.dump(p_crit_model, fh)
    with open(LAMBDA_LIMIT_PATH, "wb") as fh:
        cloudpickle.dump(lambda_limit_model, fh)
    return p_crit_model, lambda_limit_model


def load_envelope_models() -> tuple[GaussianProcessRegressor, GaussianProcessRegressor]:
    """Load both envelope GP regressors from disk."""
    if not P_CRIT_PATH.exists() or not LAMBDA_LIMIT_PATH.exists():
        raise FileNotFoundError(
            "Envelope models not found; call train_envelope_models() first."
        )
    with open(P_CRIT_PATH, "rb") as fh:
        p_crit_model = cloudpickle.load(fh)
    with open(LAMBDA_LIMIT_PATH, "rb") as fh:
        lambda_limit_model = cloudpickle.load(fh)
    return p_crit_model, lambda_limit_model


def _gp_predict(model: GaussianProcessRegressor, alpha: Any, alpha_p: Any) -> Any:
    a_arr = np.asarray(alpha, dtype=float)
    ap_arr = np.asarray(alpha_p, dtype=float)
    out_shape = np.broadcast_shapes(a_arr.shape, ap_arr.shape)
    a_b = np.broadcast_to(a_arr, out_shape).reshape(-1, 1)
    ap_b = np.broadcast_to(ap_arr, out_shape).reshape(-1, 1)
    features = np.hstack([a_b, ap_b])
    preds = model.predict(features)
    if out_shape == ():
        return float(preds[0])
    return preds.reshape(out_shape)


def predict_p_crit(model: GaussianProcessRegressor, alpha: Any, alpha_p: Any) -> Any:
    """Predict ``p_crit`` at ``(alpha, alpha_p)``. Extrapolates outside the trained envelope."""
    return _gp_predict(model, alpha, alpha_p)


def predict_lambda_limit(model: GaussianProcessRegressor, alpha: Any, alpha_p: Any) -> Any:
    """Predict ``lambda_0`` at the limit point. Extrapolates outside the trained envelope."""
    return _gp_predict(model, alpha, alpha_p)
