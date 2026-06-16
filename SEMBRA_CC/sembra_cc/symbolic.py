"""Symbolic core: SymPy-derived Π integrand, partials, and analytical det(H).

All derivatives are taken on the integrand expression (a function of r, alpha,
alpha_p, p, lambda_0, eta_0). The composite trapezoidal sum over a fixed
1001-point r-grid is constructed *after* lambdification, which keeps SymPy from
emitting a 1001-term flat Add and crashing the Python C-stack on parse.
"""

from __future__ import annotations

from typing import Callable

import cloudpickle
import numpy as np
import sympy as sp

from sembra_cc import TRAINED_MODELS_DIR

SYMBOLIC_VERSION = "1.0.0"
N_GRID = 1001
_DR = 1.0 / (N_GRID - 1)

_R_GRID = np.linspace(0.0, 1.0, N_GRID)
_WEIGHTS = np.full(N_GRID, _DR)
_WEIGHTS[0] = _DR / 2.0
_WEIGHTS[-1] = _DR / 2.0

_CACHE_PATH = TRAINED_MODELS_DIR / "symbolic_cache.pkl"


def _build_integrand():
    """Return (symbolic integrand, tuple of symbols (r, alpha, alpha_p, p, lambda_0, eta_0))."""
    r, alpha, alpha_p, p, lambda_0, eta_0 = sp.symbols(
        "r alpha alpha_p p lambda_0 eta_0", real=True
    )

    lambda_2 = r**2 + lambda_0 * (1 - r**2)
    dlambda2_dr = 2 * r * (1 - lambda_0)
    deta_dr = -2 * eta_0 * r
    inner = lambda_2 + r * dlambda2_dr
    lambda_1 = sp.sqrt(inner**2 + deta_dr**2)
    lambda_3 = 1 / (lambda_1 * lambda_2)

    u_integrand = (1 / alpha) * (
        lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3
    ) * r
    wecc_integrand = (alpha_p * lambda_3**2 / (2 * alpha)) * r
    wp_integrand = -sp.Rational(1, 2) * p * (r * lambda_2) ** 2 * deta_dr

    integrand = u_integrand + wecc_integrand - wp_integrand
    return integrand, (r, alpha, alpha_p, p, lambda_0, eta_0)


def _wrap_integrand_callable(integrand_func: Callable) -> Callable:
    """Wrap an integrand callable f(r, alpha, alpha_p, p, lambda_0, eta_0)
    into a function g(alpha, alpha_p, p, lambda_0, eta_0) that performs the
    1001-point composite trapezoidal sum over r.

    Broadcasting: scalar inputs return Python float; array inputs return an
    array of the broadcast shape (without the r axis).
    """

    def evaluate(alpha, alpha_p, p, lambda_0, eta_0):
        a = np.asarray(alpha, dtype=np.float64)
        ap = np.asarray(alpha_p, dtype=np.float64)
        pp = np.asarray(p, dtype=np.float64)
        l0 = np.asarray(lambda_0, dtype=np.float64)
        e0 = np.asarray(eta_0, dtype=np.float64)

        broadcast_shape = np.broadcast_shapes(
            a.shape, ap.shape, pp.shape, l0.shape, e0.shape
        )
        all_scalar = broadcast_shape == ()

        a_b = np.broadcast_to(a, broadcast_shape)[..., np.newaxis]
        ap_b = np.broadcast_to(ap, broadcast_shape)[..., np.newaxis]
        pp_b = np.broadcast_to(pp, broadcast_shape)[..., np.newaxis]
        l0_b = np.broadcast_to(l0, broadcast_shape)[..., np.newaxis]
        e0_b = np.broadcast_to(e0, broadcast_shape)[..., np.newaxis]

        r_b = _R_GRID

        values = integrand_func(r_b, a_b, ap_b, pp_b, l0_b, e0_b)
        # Some integrands may be independent of r (broadcast scalar across grid);
        # ensure we have a length-N_GRID last axis.
        values = np.broadcast_to(values, broadcast_shape + (N_GRID,))

        result = np.sum(values * _WEIGHTS, axis=-1)

        if all_scalar:
            return float(result)
        return result

    return evaluate


def _build_callables() -> dict[str, Callable]:
    integrand, syms = _build_integrand()
    r, alpha, alpha_p, p, lambda_0, eta_0 = syms

    # Integrand-level derivatives.
    d_l0 = sp.diff(integrand, lambda_0)
    d_e0 = sp.diff(integrand, eta_0)
    d2_l0 = sp.diff(integrand, lambda_0, 2)
    d2_e0 = sp.diff(integrand, eta_0, 2)
    d2_mixed = sp.diff(integrand, lambda_0, eta_0)

    args = (r, alpha, alpha_p, p, lambda_0, eta_0)

    f_pi = sp.lambdify(args, integrand, modules="numpy", cse=True)
    f_dl0 = sp.lambdify(args, d_l0, modules="numpy", cse=True)
    f_de0 = sp.lambdify(args, d_e0, modules="numpy", cse=True)
    f_d2l0 = sp.lambdify(args, d2_l0, modules="numpy", cse=True)
    f_d2e0 = sp.lambdify(args, d2_e0, modules="numpy", cse=True)
    f_dmix = sp.lambdify(args, d2_mixed, modules="numpy", cse=True)

    Pi = _wrap_integrand_callable(f_pi)
    dPi_dlambda_0 = _wrap_integrand_callable(f_dl0)
    dPi_deta_0 = _wrap_integrand_callable(f_de0)
    d2Pi_dl0 = _wrap_integrand_callable(f_d2l0)
    d2Pi_de0 = _wrap_integrand_callable(f_d2e0)
    d2Pi_dmix = _wrap_integrand_callable(f_dmix)

    def det_H_analytical(alpha, alpha_p, p, lambda_0, eta_0):
        a = d2Pi_dl0(alpha, alpha_p, p, lambda_0, eta_0)
        b = d2Pi_de0(alpha, alpha_p, p, lambda_0, eta_0)
        c = d2Pi_dmix(alpha, alpha_p, p, lambda_0, eta_0)
        return a * b - c * c

    return {
        "Pi": Pi,
        "dPi_dlambda_0": dPi_dlambda_0,
        "dPi_deta_0": dPi_deta_0,
        "det_H_analytical": det_H_analytical,
    }


def derive_symbolic_expressions(use_cache: bool = True) -> dict[str, Callable]:
    """Build (or load from cache) the four symbolic callables."""
    if use_cache and _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as fh:
                payload = cloudpickle.load(fh)
            if (
                isinstance(payload, dict)
                and payload.get("version") == SYMBOLIC_VERSION
                and "callables" in payload
            ):
                return payload["callables"]
        except Exception:
            pass  # fall through to rebuild

    callables = _build_callables()
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CACHE_PATH, "wb") as fh:
            cloudpickle.dump(
                {"version": SYMBOLIC_VERSION, "callables": callables}, fh
            )
    except Exception:
        pass
    return callables
