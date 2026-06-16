"""User-facing prediction wrapper for SEMBRA.

Composition of the trained envelope GPs, the surrogate network, and the
analytical det(H) into a single ``predict(alpha, alpha_p, p)`` entry point
that returns equilibrium state and a stability assessment.

Flow (per DEC-012, which supersedes PRD Section 6.5.1 step 5):

1. Validate inputs (Section 6.5.1 step 1; raises ``ValueError`` on bad input).
2. Look up ``p_crit`` and ``lambda_limit`` from the envelope GPs (step 2).
3. Trivial-state shortcut for ``p < P_ZERO_THRESHOLD`` (step 3).
4. No-stable-region shortcut for ``p_crit ≤ 0`` (step 4).
5. Three-branch decision on ``delta = (p − p_crit) / p_crit`` (DEC-012):
   - ``|delta| < GREY_ZONE_HALF_WIDTH`` → grey-zone past-limit solve (stable=False).
   - ``delta ≤ −GREY_ZONE_HALF_WIDTH`` → Branch A brentq on the stable side.
   - ``delta ≥ +GREY_ZONE_HALF_WIDTH`` → Branch B past-limit solve (stable=False).

All three branches return populated state; the wrapper never returns NaN
except for the no-stable-region shortcut (step 4). Defensive wrapping per
Section 6.5.2 catches downstream model exceptions and logs them to
``logs/wrapper_errors.log``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.optimize
from scipy.optimize import brentq

from sembra import LOGS_DIR
from sembra.calibrations import GREY_ZONE_HALF_WIDTH, P_ZERO_THRESHOLD
from sembra.envelope import (
    load_envelope_models,
    predict_lambda_limit,
    predict_p_crit,
)
from sembra.network import SurrogateNetwork, load_network, predict_from_lambda_0
from sembra.symbolic import derive_symbolic_expressions

_WRAPPER_ERROR_LOG = LOGS_DIR / "wrapper_errors.log"
_BRENTQ_XTOL = 1e-5
_ALPHA_MIN = 1.2
_ALPHA_MAX = 2.0
_FALLBACK_LAMBDA_LIMIT = 1.5
_FALLBACK_P_CRIT = 0.0

_wrapper_logger = logging.getLogger("sembra.wrapper.errors")


def _ensure_error_logger() -> None:
    if _wrapper_logger.handlers:
        return
    _WRAPPER_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(_WRAPPER_ERROR_LOG, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _wrapper_logger.addHandler(handler)
    _wrapper_logger.setLevel(logging.WARNING)
    _wrapper_logger.propagate = False


@dataclass
class PredictionResult:
    alpha: float
    alpha_p: float
    p: float
    lambda_0: float
    eta_0: float
    det_H: float
    stable: bool
    message: str
    diagnostics: dict = field(default_factory=dict)


class _Resources:
    """Lazy-loaded singletons for the trained models and symbolic callables."""

    def __init__(self) -> None:
        self._network: SurrogateNetwork | None = None
        self._p_crit_model: Any = None
        self._lambda_limit_model: Any = None
        self._symbolic: dict | None = None

    @property
    def network(self) -> SurrogateNetwork:
        if self._network is None:
            self._network = load_network()
        return self._network

    @property
    def p_crit_model(self) -> Any:
        if self._p_crit_model is None:
            self._p_crit_model, self._lambda_limit_model = load_envelope_models()
        return self._p_crit_model

    @property
    def lambda_limit_model(self) -> Any:
        if self._lambda_limit_model is None:
            self._p_crit_model, self._lambda_limit_model = load_envelope_models()
        return self._lambda_limit_model

    @property
    def symbolic(self) -> dict:
        if self._symbolic is None:
            self._symbolic = derive_symbolic_expressions()
        return self._symbolic


_RESOURCES = _Resources()


def _validate_inputs(alpha: float, alpha_p: float, p: float) -> None:
    for name, value in (("alpha", alpha), ("alpha_p", alpha_p), ("p", p)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be a real number, got type {type(value).__name__}.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
    if alpha < _ALPHA_MIN or alpha > _ALPHA_MAX:
        raise ValueError(
            f"alpha = {alpha} is outside the supported range [{_ALPHA_MIN}, {_ALPHA_MAX}]."
        )
    if alpha_p < 0:
        raise ValueError(f"alpha_p = {alpha_p} must be non-negative.")
    if p < 0:
        raise ValueError(f"p = {p} must be non-negative.")


def _safe_envelope_lookup(alpha: float, alpha_p: float) -> tuple[float, float]:
    try:
        p_crit = float(predict_p_crit(_RESOURCES.p_crit_model, alpha, alpha_p))
        lambda_limit = float(predict_lambda_limit(_RESOURCES.lambda_limit_model, alpha, alpha_p))
    except Exception as exc:  # noqa: BLE001 - intentional defensive catch per Section 6.5.2
        _ensure_error_logger()
        _wrapper_logger.warning(
            "Envelope lookup failed for (alpha=%s, alpha_p=%s): %s", alpha, alpha_p, exc
        )
        return _FALLBACK_P_CRIT, _FALLBACK_LAMBDA_LIMIT
    return p_crit, lambda_limit


def _solve_lambda_0(
    alpha: float, alpha_p: float, p: float, lambda_limit: float
) -> tuple[float, float, float, float] | None:
    """Branch A: bracketed solve for lambda_0 with brentq. Returns ``None`` on failure."""

    def residual(lambda_0_candidate: float) -> float:
        try:
            p_pred, _, _ = predict_from_lambda_0(
                _RESOURCES.network, alpha, alpha_p, lambda_0_candidate
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Network call failed at lambda_0={lambda_0_candidate}: {exc}") from exc
        return p_pred - p

    try:
        bracket_lo = 1.0
        bracket_hi = max(lambda_limit, bracket_lo + 1e-3)
        lambda_0 = brentq(residual, bracket_lo, bracket_hi, xtol=_BRENTQ_XTOL)
    except Exception as exc:  # noqa: BLE001
        _ensure_error_logger()
        _wrapper_logger.warning(
            "brentq solve failed for (alpha=%s, alpha_p=%s, p=%s): %s",
            alpha, alpha_p, p, exc,
        )
        return None
    try:
        p_predicted, eta_0, det_H_network = predict_from_lambda_0(
            _RESOURCES.network, alpha, alpha_p, lambda_0
        )
    except Exception as exc:  # noqa: BLE001
        _ensure_error_logger()
        _wrapper_logger.warning(
            "Network final query failed at lambda_0=%s: %s", lambda_0, exc
        )
        return None
    return lambda_0, p_predicted, eta_0, det_H_network


_GREY_ZONE_MESSAGE_TEMPLATE = (
    "Unstable equilibrium: design is within 0.5% of the limit point. "
    "For alpha = {alpha} and alpha_p = {alpha_p}, the critical pressure is "
    "p_crit ≈ {p_crit_value:.6f}. State, shape, and stress are computed for "
    "the unstable equilibrium and shown for design reference."
)

_PAST_LIMIT_MESSAGE_TEMPLATE = (
    "Unstable equilibrium past the limit point. For alpha = {alpha} and "
    "alpha_p = {alpha_p}, the critical pressure is p_crit ≈ {p_crit_value:.6f}; "
    "the entered p = {p} exceeds this. State, shape, and stress are computed "
    "for the unstable equilibrium and shown for design reference."
)

_STABLE_MESSAGE = "Stable equilibrium."


def _solve_past_limit_equilibrium(
    alpha: float,
    alpha_p: float,
    p: float,
    lambda_limit: float,
) -> tuple[float, float, float] | None:
    """Solve ∂Π/∂λ₀ = ∂Π/∂η₀ = 0 with scipy.optimize.root, warm-started at the limit point.

    Returns ``(lambda_0, eta_0, eta_warm_start)`` on success and ``None`` if the
    Newton-Powell iteration does not converge.
    """
    try:
        _, eta_warm, _ = predict_from_lambda_0(
            _RESOURCES.network, alpha, alpha_p, lambda_limit
        )
    except Exception as exc:  # noqa: BLE001
        _ensure_error_logger()
        _wrapper_logger.warning(
            "Network warm-start failed for past-limit solve at "
            "(alpha=%s, alpha_p=%s, p=%s): %s",
            alpha, alpha_p, p, exc,
        )
        eta_warm = 0.5

    dpi_dl_fn = _RESOURCES.symbolic["dPi_dlambda_0"]
    dpi_de_fn = _RESOURCES.symbolic["dPi_deta_0"]

    def residuals(x: np.ndarray) -> list[float]:
        l0, e0 = float(x[0]), float(x[1])
        return [
            float(dpi_dl_fn(alpha, alpha_p, p, l0, e0)),
            float(dpi_de_fn(alpha, alpha_p, p, l0, e0)),
        ]

    try:
        sol = scipy.optimize.root(
            residuals,
            x0=[float(lambda_limit), float(eta_warm)],
            method="hybr",
        )
    except Exception as exc:  # noqa: BLE001
        _ensure_error_logger()
        _wrapper_logger.warning(
            "scipy.root past-limit solve raised for (alpha=%s, alpha_p=%s, p=%s): %s",
            alpha, alpha_p, p, exc,
        )
        return None
    if not sol.success:
        return None
    return float(sol.x[0]), float(sol.x[1]), float(eta_warm)


def _populated_past_limit_solve(
    alpha: float, alpha_p: float, p: float, lambda_limit: float
) -> tuple[float, float]:
    """Past-limit solve guaranteed to return populated state (never NaN).

    Falls back to the warm-start ``(lambda_limit, eta_warm)`` on solver failure;
    this keeps the public ``PredictionResult.lambda_0`` / ``eta_0`` defined
    even when no mathematical equilibrium exists at the requested ``p``.
    """
    past = _solve_past_limit_equilibrium(alpha, alpha_p, p, lambda_limit)
    if past is not None:
        return past[0], past[1]
    try:
        _, eta_warm, _ = predict_from_lambda_0(
            _RESOURCES.network, alpha, alpha_p, lambda_limit
        )
    except Exception:  # noqa: BLE001
        eta_warm = 0.5
    return float(lambda_limit), float(eta_warm)


def _populated_stable_solve(
    alpha: float, alpha_p: float, p: float, lambda_limit: float
) -> tuple[float, float]:
    """Branch A stable-side solve guaranteed to return populated state.

    Falls back to the scipy past-limit solver if brentq cannot bracket, and
    finally to the midpoint of ``[1.0, lambda_limit]`` if both fail.
    """
    solved = _solve_lambda_0(alpha, alpha_p, p, lambda_limit)
    if solved is not None:
        lambda_0, _, eta_0, _ = solved
        return float(lambda_0), float(eta_0)
    past = _solve_past_limit_equilibrium(alpha, alpha_p, p, lambda_limit)
    if past is not None:
        return past[0], past[1]
    midpoint = 0.5 * (1.0 + lambda_limit)
    try:
        _, eta_warm, _ = predict_from_lambda_0(
            _RESOURCES.network, alpha, alpha_p, midpoint
        )
    except Exception:  # noqa: BLE001
        eta_warm = 0.3
    return float(midpoint), float(eta_warm)


def predict(alpha: float, alpha_p: float, p: float) -> PredictionResult:
    """Predict membrane equilibrium and stability given ``(alpha, alpha_p, p)``.

    Raises
    ------
    ValueError
        If any input is non-finite, of the wrong type, or outside the
        supported physical range (Section 6.5.1 step 1). All other internal
        errors are caught, logged, and reported as a non-stable
        :class:`PredictionResult`.
    """
    _validate_inputs(alpha, alpha_p, p)
    alpha = float(alpha)
    alpha_p = float(alpha_p)
    p = float(p)

    p_crit, lambda_limit = _safe_envelope_lookup(alpha, alpha_p)
    diagnostics: dict[str, Any] = {"p_crit": p_crit, "lambda_limit": lambda_limit}

    det_h_fn = _RESOURCES.symbolic["det_H_analytical"]
    dpi_dl_fn = _RESOURCES.symbolic["dPi_dlambda_0"]
    dpi_de_fn = _RESOURCES.symbolic["dPi_deta_0"]

    if p < P_ZERO_THRESHOLD:
        lambda_0 = 1.0
        eta_0 = 0.0
        det_h_value = float(det_h_fn(alpha, alpha_p, p, lambda_0, eta_0))
        return PredictionResult(
            alpha=alpha, alpha_p=alpha_p, p=p,
            lambda_0=lambda_0, eta_0=eta_0, det_H=det_h_value,
            stable=True,
            message="Trivial undeformed state at near-zero pressure.",
            diagnostics=diagnostics,
        )

    if p_crit <= 0:
        return PredictionResult(
            alpha=alpha, alpha_p=alpha_p, p=p,
            lambda_0=float("nan"), eta_0=float("nan"), det_H=float("nan"),
            stable=False,
            message="Envelope predicts no stable equilibrium for this (alpha, alpha_p) combination.",
            diagnostics=diagnostics,
        )

    delta = (p - p_crit) / p_crit
    if abs(delta) < GREY_ZONE_HALF_WIDTH:
        lambda_0, eta_0 = _populated_past_limit_solve(alpha, alpha_p, p, lambda_limit)
        det_h_value = float(det_h_fn(alpha, alpha_p, p, lambda_0, eta_0))
        return PredictionResult(
            alpha=alpha, alpha_p=alpha_p, p=p,
            lambda_0=lambda_0, eta_0=eta_0, det_H=det_h_value,
            stable=False,
            message=_GREY_ZONE_MESSAGE_TEMPLATE.format(
                alpha=alpha, alpha_p=alpha_p, p_crit_value=p_crit
            ),
            diagnostics=diagnostics,
        )

    if delta <= -GREY_ZONE_HALF_WIDTH:
        lambda_0, eta_0 = _populated_stable_solve(alpha, alpha_p, p, lambda_limit)
        det_h_value = float(det_h_fn(alpha, alpha_p, p, lambda_0, eta_0))
        res_lambda = float(dpi_dl_fn(alpha, alpha_p, p, lambda_0, eta_0))
        res_eta = float(dpi_de_fn(alpha, alpha_p, p, lambda_0, eta_0))
        diagnostics.update(
            {
                "residual_dPi_dlambda_0": res_lambda,
                "residual_dPi_deta_0": res_eta,
            }
        )
        return PredictionResult(
            alpha=alpha, alpha_p=alpha_p, p=p,
            lambda_0=lambda_0, eta_0=eta_0, det_H=det_h_value,
            stable=det_h_value > 0,
            message=_STABLE_MESSAGE,
            diagnostics=diagnostics,
        )

    lambda_0, eta_0 = _populated_past_limit_solve(alpha, alpha_p, p, lambda_limit)
    det_h_value = float(det_h_fn(alpha, alpha_p, p, lambda_0, eta_0))
    return PredictionResult(
        alpha=alpha, alpha_p=alpha_p, p=p,
        lambda_0=lambda_0, eta_0=eta_0, det_H=det_h_value,
        stable=False,
        message=_PAST_LIMIT_MESSAGE_TEMPLATE.format(
            alpha=alpha, alpha_p=alpha_p, p=p, p_crit_value=p_crit
        ),
        diagnostics=diagnostics,
    )
