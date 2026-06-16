"""User-facing `predict(alpha, alpha_p, p) -> PredictionResult` entry point.

Routes the user's query through input validation, envelope lookup, and one of
three branches (trivial, stable, grey-zone, past-limit). The det_H field of
the returned PredictionResult is always sourced from the analytical Hessian
in symbolic.py.
"""

from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.optimize
import torch

from sembra_cc import LOGS_DIR
from sembra_cc.calibrations import GREY_ZONE_HALF_WIDTH, P_ZERO_THRESHOLD
from sembra_cc.envelope import load_envelope_models, predict_lambda_limit, predict_p_crit
from sembra_cc.network import load_network, predict_from_lambda_0
from sembra_cc.symbolic import derive_symbolic_expressions

_BRENTQ_XTOL = 1e-5
_ALPHA_MIN = 1.2
_ALPHA_MAX = 2.0
_FALLBACK_LAMBDA_LIMIT = 1.5
_FALLBACK_P_CRIT = 0.0
_WRAPPER_ERROR_LOG = LOGS_DIR / "wrapper_errors.log"


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


def _ensure_logger() -> logging.Logger:
    logger = logging.getLogger("sembra_cc.wrapper")
    if not logger.handlers:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_WRAPPER_ERROR_LOG)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
    return logger


def _log_warning(context: str, exc: BaseException) -> None:
    logger = _ensure_logger()
    logger.warning("%s: %s\n%s", context, exc, traceback.format_exc())


class _Resources:
    """Lazy-load network, envelope GPs, and symbolic callables once per process."""

    _instance: Optional["_Resources"] = None

    def __init__(self):
        self._network = None
        self._gp_p = None
        self._gp_l = None
        self._symbolic = None

    @classmethod
    def instance(cls) -> "_Resources":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def network(self):
        if self._network is None:
            try:
                self._network = load_network()
            except Exception as e:  # noqa
                _log_warning("network load failed", e)
                raise
        return self._network

    @property
    def envelopes(self):
        if self._gp_p is None or self._gp_l is None:
            try:
                self._gp_p, self._gp_l = load_envelope_models()
            except Exception as e:  # noqa
                _log_warning("envelope load failed", e)
                raise
        return self._gp_p, self._gp_l

    @property
    def symbolic(self):
        if self._symbolic is None:
            try:
                self._symbolic = derive_symbolic_expressions()
            except Exception as e:  # noqa
                _log_warning("symbolic load failed", e)
                raise RuntimeError(f"symbolic core unloadable: {e}")
        return self._symbolic


def _validate_inputs(alpha, alpha_p, p) -> None:
    for name, value in (("alpha", alpha), ("alpha_p", alpha_p), ("p", p)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be int or float, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if alpha < _ALPHA_MIN or alpha > _ALPHA_MAX:
        raise ValueError(
            f"alpha={alpha} outside allowed range [{_ALPHA_MIN}, {_ALPHA_MAX}]"
        )
    if alpha_p < 0:
        raise ValueError(f"alpha_p={alpha_p} must be >= 0")
    if p < 0:
        raise ValueError(f"p={p} must be >= 0")


def _envelope_lookup(alpha: float, alpha_p: float) -> tuple[float, float]:
    res = _Resources.instance()
    try:
        gp_p, gp_l = res.envelopes
        p_crit = float(predict_p_crit(gp_p, alpha, alpha_p))
        lambda_limit = float(predict_lambda_limit(gp_l, alpha, alpha_p))
        return p_crit, lambda_limit
    except Exception as e:  # noqa
        _log_warning("envelope lookup failed", e)
        return _FALLBACK_P_CRIT, _FALLBACK_LAMBDA_LIMIT


def _safe_det_H(alpha: float, alpha_p: float, p: float, l0: float, e0: float) -> float:
    funcs = _Resources.instance().symbolic
    return float(funcs["det_H_analytical"](alpha, alpha_p, p, l0, e0))


def _safe_residuals(alpha: float, alpha_p: float, p: float, l0: float, e0: float):
    funcs = _Resources.instance().symbolic
    return (
        float(funcs["dPi_dlambda_0"](alpha, alpha_p, p, l0, e0)),
        float(funcs["dPi_deta_0"](alpha, alpha_p, p, l0, e0)),
    )


def _query_network_eta_warm(alpha: float, alpha_p: float, lambda_limit: float) -> float:
    res = _Resources.instance()
    try:
        _, eta_warm, _ = predict_from_lambda_0(res.network, alpha, alpha_p, lambda_limit)
        if not math.isfinite(eta_warm):
            return 0.5
        return float(eta_warm)
    except Exception as e:  # noqa
        _log_warning("network warm-start failed", e)
        return 0.5


def _solve_past_limit_equilibrium(
    alpha: float, alpha_p: float, p: float, lambda_limit: float
) -> Optional[tuple[float, float, float]]:
    eta_warm = _query_network_eta_warm(alpha, alpha_p, lambda_limit)
    funcs = _Resources.instance().symbolic

    def residuals(x):
        l0, e0 = float(x[0]), float(x[1])
        return [
            float(funcs["dPi_dlambda_0"](alpha, alpha_p, p, l0, e0)),
            float(funcs["dPi_deta_0"](alpha, alpha_p, p, l0, e0)),
        ]

    try:
        sol = scipy.optimize.root(
            residuals, x0=[lambda_limit, eta_warm], method="hybr"
        )
    except Exception as e:  # noqa
        _log_warning("past-limit root solve raised", e)
        return None

    if not sol.success:
        return None
    return float(sol.x[0]), float(sol.x[1]), float(eta_warm)


def _populated_past_limit_solve(
    alpha: float, alpha_p: float, p: float, lambda_limit: float
) -> tuple[float, float]:
    sol = _solve_past_limit_equilibrium(alpha, alpha_p, p, lambda_limit)
    if sol is not None:
        return sol[0], sol[1]
    eta_warm = _query_network_eta_warm(alpha, alpha_p, lambda_limit)
    return float(lambda_limit), float(eta_warm)


def _solve_lambda_0(
    alpha: float, alpha_p: float, p_target: float, lambda_limit: float
) -> Optional[tuple[float, float, float, float]]:
    res = _Resources.instance()
    bracket_lo = 1.0
    bracket_hi = max(lambda_limit, 1.001)

    def residual(l0):
        try:
            p_pred, _, _ = predict_from_lambda_0(res.network, alpha, alpha_p, float(l0))
        except Exception as e:  # noqa
            _log_warning("network call inside brentq failed", e)
            raise
        return float(p_pred) - p_target

    try:
        l0_solved = scipy.optimize.brentq(
            residual, bracket_lo, bracket_hi, xtol=_BRENTQ_XTOL
        )
    except Exception as e:  # noqa
        _log_warning("brentq stable-branch solve failed", e)
        return None

    try:
        p_pred, eta_0, det_H_net = predict_from_lambda_0(
            res.network, alpha, alpha_p, float(l0_solved)
        )
    except Exception as e:  # noqa
        _log_warning("network final query failed", e)
        return None

    return float(l0_solved), float(p_pred), float(eta_0), float(det_H_net)


def _populated_stable_solve(
    alpha: float, alpha_p: float, p: float, lambda_limit: float
) -> tuple[float, float, dict]:
    diagnostics_extra: dict = {}
    solved = _solve_lambda_0(alpha, alpha_p, p, lambda_limit)
    if solved is not None:
        l0, p_pred, eta_0, det_H_net = solved
        diagnostics_extra["network_p_pred"] = p_pred
        diagnostics_extra["network_det_H"] = det_H_net
        return l0, eta_0, diagnostics_extra

    past = _solve_past_limit_equilibrium(alpha, alpha_p, p, lambda_limit)
    if past is not None:
        diagnostics_extra["fallback"] = "past_limit_root"
        return past[0], past[1], diagnostics_extra

    diagnostics_extra["fallback"] = "midpoint_default"
    midpoint = 0.5 * (1.0 + lambda_limit)
    eta_warm = _query_network_eta_warm(alpha, alpha_p, lambda_limit)
    return float(midpoint), float(eta_warm) if eta_warm > 0 else 0.3, diagnostics_extra


def predict(alpha: float, alpha_p: float, p: float) -> PredictionResult:
    """Predict the equilibrium state at (alpha, alpha_p, p)."""
    _validate_inputs(alpha, alpha_p, p)

    p_crit, lambda_limit = _envelope_lookup(float(alpha), float(alpha_p))
    diagnostics: dict = {"p_crit": p_crit, "lambda_limit": lambda_limit}

    # Step 3: trivial-state shortcut.
    if p < P_ZERO_THRESHOLD:
        l0, e0 = 1.0, 0.0
        det_H = _safe_det_H(alpha, alpha_p, p, l0, e0)
        return PredictionResult(
            alpha=float(alpha),
            alpha_p=float(alpha_p),
            p=float(p),
            lambda_0=l0,
            eta_0=e0,
            det_H=det_H,
            stable=True,
            message="Trivial undeformed state at near-zero pressure.",
            diagnostics=diagnostics,
        )

    # Step 4: no stable region at all.
    if p_crit <= 0:
        return PredictionResult(
            alpha=float(alpha),
            alpha_p=float(alpha_p),
            p=float(p),
            lambda_0=float("nan"),
            eta_0=float("nan"),
            det_H=float("nan"),
            stable=False,
            message="Envelope predicts no stable equilibrium for this (alpha, alpha_p) combination.",
            diagnostics=diagnostics,
        )

    # Step 5: three-branch decision.
    delta = (p - p_crit) / p_crit

    if abs(delta) < GREY_ZONE_HALF_WIDTH:
        l0, e0 = _populated_past_limit_solve(alpha, alpha_p, p, lambda_limit)
        det_H = _safe_det_H(alpha, alpha_p, p, l0, e0)
        msg = (
            f"Unstable equilibrium: design is within 0.5% of the limit point. "
            f"For alpha = {alpha} and alpha_p = {alpha_p}, the critical pressure is "
            f"p_crit ≈ {p_crit:.6f}. State, shape, and stress are computed for "
            f"the unstable equilibrium and shown for design reference."
        )
        return PredictionResult(
            alpha=float(alpha),
            alpha_p=float(alpha_p),
            p=float(p),
            lambda_0=l0,
            eta_0=e0,
            det_H=det_H,
            stable=False,
            message=msg,
            diagnostics=diagnostics,
        )

    if delta <= -GREY_ZONE_HALF_WIDTH:
        l0, e0, extra = _populated_stable_solve(alpha, alpha_p, p, lambda_limit)
        det_H = _safe_det_H(alpha, alpha_p, p, l0, e0)
        try:
            r_l0, r_e0 = _safe_residuals(alpha, alpha_p, p, l0, e0)
        except Exception:  # noqa
            r_l0, r_e0 = float("nan"), float("nan")
        diagnostics["dPi_dlambda_0"] = r_l0
        diagnostics["dPi_deta_0"] = r_e0
        diagnostics.update(extra)
        return PredictionResult(
            alpha=float(alpha),
            alpha_p=float(alpha_p),
            p=float(p),
            lambda_0=l0,
            eta_0=e0,
            det_H=det_H,
            stable=det_H > 0,
            message="Stable equilibrium.",
            diagnostics=diagnostics,
        )

    # delta >= +GREY_ZONE_HALF_WIDTH: past-limit.
    l0, e0 = _populated_past_limit_solve(alpha, alpha_p, p, lambda_limit)
    det_H = _safe_det_H(alpha, alpha_p, p, l0, e0)
    msg = (
        f"Unstable equilibrium past the limit point. For alpha = {alpha} and "
        f"alpha_p = {alpha_p}, the critical pressure is p_crit ≈ "
        f"{p_crit:.6f}; the entered p = {p} exceeds this. State, shape, and "
        f"stress are computed for the unstable equilibrium and shown for design reference."
    )
    return PredictionResult(
        alpha=float(alpha),
        alpha_p=float(alpha_p),
        p=float(p),
        lambda_0=l0,
        eta_0=e0,
        det_H=det_H,
        stable=False,
        message=msg,
        diagnostics=diagnostics,
    )
