"""Gaussian-process stability envelopes for the charge-controlled scenario.

Three GP regressors describe boundaries in the ``(alpha, Phi)`` design space:

- ``p_crit_envelope``: critical pressure at the limit point of the ``(alpha,
  Phi)`` slice.
- ``lambda_limit_envelope``: the value of ``lambda_0`` at that limit point.
- ``wrinkling_envelope``: the wrinkling-boundary pressure ``p_wrinkle`` at which
  the minimum circumferential stress first reaches zero on the ``(alpha, Phi)``
  slice.

Each envelope is a fitted scikit-learn ``Pipeline`` of a ``StandardScaler``
(the input normalization over ``(alpha, Phi)``) followed by a
``GaussianProcessRegressor``. The scaler carries the normalization parameters
with the model, so this module applies the identical transform at inference with
no extra bookkeeping. Fitting happens in the workshop training scripts; this
module only loads the pre-trained artifacts and evaluates them.

``load_envelopes()`` keeps its original two-tuple contract (``p_crit``,
``lambda_limit``) so existing callers are unchanged; the wrinkling envelope is
loaded via ``load_wrinkling_envelope()`` and all three are available together
from ``load_all_envelopes()``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

# Locations of the shipped GP artifacts inside the package.
_MODELS_DIR = Path(__file__).resolve().parent / "trained_models"
_P_CRIT_PATH = _MODELS_DIR / "p_crit_envelope.pkl"
_LAMBDA_LIMIT_PATH = _MODELS_DIR / "lambda_limit_envelope.pkl"
_WRINKLING_PATH = _MODELS_DIR / "wrinkling_envelope.pkl"

# Caches for the loaded envelopes, populated lazily on first load.
_ENVELOPE_CACHE: tuple[Pipeline, Pipeline] | None = None
_WRINKLING_CACHE: Pipeline | None = None


def _load_pickle(path: Path) -> Pipeline:
    """Load a single pickled envelope pipeline, with a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"GP envelope artifact not found at {path}. Fit the envelopes first "
            "with the training script before calling load_envelopes()."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_envelopes() -> tuple[Pipeline, Pipeline]:
    """Load both fitted GP envelopes from the package's ``trained_models``.

    Returns ``(p_crit_envelope, lambda_limit_envelope)``. The result is cached,
    so repeated calls return the same objects.
    """
    global _ENVELOPE_CACHE
    if _ENVELOPE_CACHE is None:
        _ENVELOPE_CACHE = (
            _load_pickle(_P_CRIT_PATH),
            _load_pickle(_LAMBDA_LIMIT_PATH),
        )
    return _ENVELOPE_CACHE


def load_wrinkling_envelope() -> Pipeline:
    """Load the fitted wrinkling-boundary GP envelope (cached).

    Kept separate from :func:`load_envelopes` so that function's two-tuple
    contract stays intact for existing callers. Repeated calls return the same
    object.
    """
    global _WRINKLING_CACHE
    if _WRINKLING_CACHE is None:
        _WRINKLING_CACHE = _load_pickle(_WRINKLING_PATH)
    return _WRINKLING_CACHE


def load_all_envelopes() -> tuple[Pipeline, Pipeline, Pipeline]:
    """Load all three envelopes as ``(p_crit, lambda_limit, wrinkling)`` (cached)."""
    p_crit, lambda_limit = load_envelopes()
    return p_crit, lambda_limit, load_wrinkling_envelope()


def _predict(envelope: Pipeline, alpha, Phi):
    """Evaluate an envelope at ``(alpha, Phi)``.

    Accepts scalars or array-likes. Returns a Python ``float`` for scalar input
    and a NumPy array for array input (broadcast to a common shape).
    """
    alpha_arr = np.asarray(alpha, dtype=float)
    phi_arr = np.asarray(Phi, dtype=float)
    scalar_input = alpha_arr.ndim == 0 and phi_arr.ndim == 0

    alpha_b, phi_b = np.broadcast_arrays(alpha_arr, phi_arr)
    features = np.column_stack([alpha_b.ravel(), phi_b.ravel()])
    preds = envelope.predict(features)

    if scalar_input:
        return float(preds[0])
    return preds.reshape(alpha_b.shape)


def predict_p_crit(envelope: Pipeline, alpha, Phi):
    """Predict the critical pressure ``p_crit`` at ``(alpha, Phi)``.

    ``envelope`` is the p_crit pipeline from :func:`load_envelopes`. Scalar in ->
    scalar out; array in -> array out.
    """
    return _predict(envelope, alpha, Phi)


def predict_lambda_limit(envelope: Pipeline, alpha, Phi):
    """Predict the limit-point stretch ``lambda_limit`` at ``(alpha, Phi)``.

    ``envelope`` is the lambda_limit pipeline from :func:`load_envelopes`. Scalar
    in -> scalar out; array in -> array out.
    """
    return _predict(envelope, alpha, Phi)


def predict_p_wrinkle(alpha, Phi):
    """Predict the wrinkling-boundary pressure ``p_wrinkle`` at ``(alpha, Phi)``.

    Self-contained: loads its own cached wrinkling envelope, so no envelope
    argument is needed (unlike :func:`predict_p_crit`). Broadcasting behavior
    matches the other predictors: scalar in -> scalar out; array in -> array out,
    with a scalar ``alpha`` broadcast against an array ``Phi``.
    """
    return _predict(load_wrinkling_envelope(), alpha, Phi)
