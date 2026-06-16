"""Symbolic derivation of the total potential energy Pi and its derivatives.

Implements the physics of PRD Section 4 (energy terms, stretches, Hessian)
and the composite-trapezoidal integration of Section 4.5 with the prescribed
1001-point uniform grid in r in [0, 1].

The construction differs from a literal reading of Section 4.7 in one
implementation detail recorded as DEC-002: differentiation w.r.t.
(lambda_0, eta_0) is performed at the SymPy integrand level rather than on
the post-summation Pi expression. Because the trapezoidal sum is a finite
linear combination and r is independent of (lambda_0, eta_0), the two
constructions yield numerically identical callables, while the
integrand-level approach avoids a Python C-stack overflow when lambdify
prints a 1001-term flat ``Add`` and ``compile()`` parses it into a
left-associative AST. The required callable signature
``f(alpha, alpha_p, p, lambda_0, eta_0)`` and the prescribed trapezoidal
weights are preserved exactly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import cloudpickle
import numpy as np
import sympy as sp

from sembra import TRAINED_MODELS_DIR

SYMBOLIC_VERSION = "1.0.0"
N_GRID = 1001
_DR = 1.0 / (N_GRID - 1)
_CACHE_PATH = TRAINED_MODELS_DIR / "symbolic_cache.pkl"

_R_GRID = np.linspace(0.0, 1.0, N_GRID)
_TRAP_WEIGHTS = np.full(N_GRID, _DR)
_TRAP_WEIGHTS[0] = 0.5 * _DR
_TRAP_WEIGHTS[-1] = 0.5 * _DR

logger = logging.getLogger(__name__)


def _build_symbols() -> dict[str, sp.Symbol]:
    return {
        "alpha": sp.Symbol("alpha", positive=True),
        "alpha_p": sp.Symbol("alpha_p", nonnegative=True),
        "p": sp.Symbol("p", nonnegative=True),
        "lambda_0": sp.Symbol("lambda_0", positive=True),
        "eta_0": sp.Symbol("eta_0", nonnegative=True),
        "r": sp.Symbol("r"),
    }


def _build_integrand(syms: dict[str, sp.Symbol]) -> sp.Expr:
    alpha = syms["alpha"]
    alpha_p = syms["alpha_p"]
    p = syms["p"]
    lambda_0 = syms["lambda_0"]
    eta_0 = syms["eta_0"]
    r = syms["r"]

    lambda_2 = r**2 + lambda_0 * (1 - r**2)
    eta = eta_0 * (1 - r**2)
    dlambda2_dr = sp.diff(lambda_2, r)
    deta_dr = sp.diff(eta, r)
    lambda_1 = sp.sqrt((lambda_2 + r * dlambda2_dr) ** 2 + deta_dr**2)
    lambda_3 = 1 / (lambda_1 * lambda_2)

    u_integrand = (1 / alpha) * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3) * r
    we_integrand = (alpha_p / (2 * alpha * lambda_3**2)) * r
    wp_integrand = -sp.Rational(1, 2) * p * (r * lambda_2) ** 2 * deta_dr

    return u_integrand - we_integrand - wp_integrand


def _broadcast_inputs(
    alpha: Any, alpha_p: Any, p: Any, lambda_0: Any, eta_0: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], bool]:
    arrays = [np.asarray(x, dtype=float) for x in (alpha, alpha_p, p, lambda_0, eta_0)]
    out_shape = np.broadcast_shapes(*[a.shape for a in arrays])
    scalar = out_shape == ()
    broadcasted = [
        np.broadcast_to(a, out_shape).reshape(-1, 1) if not scalar else a.reshape(1, 1)
        for a in arrays
    ]
    return (*broadcasted, out_shape, scalar)


def _make_trap_callable(integrand_fn: Callable) -> Callable:
    """Wrap a 6-arg integrand callable into a 5-arg trapezoidal-sum callable."""
    r_row = _R_GRID.reshape(1, -1)
    weights_row = _TRAP_WEIGHTS.reshape(1, -1)

    def trap_call(alpha: Any, alpha_p: Any, p: Any, lambda_0: Any, eta_0: Any) -> Any:
        a, ap, pp, l0, e0, out_shape, scalar = _broadcast_inputs(
            alpha, alpha_p, p, lambda_0, eta_0
        )
        vals = integrand_fn(a, ap, pp, l0, e0, r_row)
        if np.ndim(vals) == 0 or (hasattr(vals, "shape") and vals.shape == ()):
            vals = np.broadcast_to(vals, (a.shape[0], r_row.shape[1]))
        elif vals.shape != (a.shape[0], r_row.shape[1]):
            vals = np.broadcast_to(vals, (a.shape[0], r_row.shape[1]))
        summed = np.sum(weights_row * vals, axis=1)
        if scalar:
            return float(summed[0])
        return summed.reshape(out_shape)

    return trap_call


def _make_det_h_callable(d2l_fn: Callable, d2e_fn: Callable, dle_fn: Callable) -> Callable:
    def det_h(alpha: Any, alpha_p: Any, p: Any, lambda_0: Any, eta_0: Any) -> Any:
        a = d2l_fn(alpha, alpha_p, p, lambda_0, eta_0)
        b = d2e_fn(alpha, alpha_p, p, lambda_0, eta_0)
        c = dle_fn(alpha, alpha_p, p, lambda_0, eta_0)
        return a * b - c**2

    return det_h


def _derive_all() -> dict[str, Callable]:
    syms = _build_symbols()
    args_with_r = [
        syms["alpha"], syms["alpha_p"], syms["p"],
        syms["lambda_0"], syms["eta_0"], syms["r"],
    ]
    lam0, eta0 = syms["lambda_0"], syms["eta_0"]

    logger.info("Building Pi integrand symbolically.")
    integrand = _build_integrand(syms)

    logger.info("Differentiating integrand (commutes with the trapezoidal sum).")
    int_dl = sp.diff(integrand, lam0)
    int_de = sp.diff(integrand, eta0)
    int_d2l = sp.diff(int_dl, lam0)
    int_d2e = sp.diff(int_de, eta0)
    int_dle = sp.diff(int_dl, eta0)

    logger.info("Lambdifying integrand-level expressions with CSE.")
    integrand_fn = sp.lambdify(args_with_r, integrand, modules="numpy", cse=True)
    dl_fn = sp.lambdify(args_with_r, int_dl, modules="numpy", cse=True)
    de_fn = sp.lambdify(args_with_r, int_de, modules="numpy", cse=True)
    d2l_fn = sp.lambdify(args_with_r, int_d2l, modules="numpy", cse=True)
    d2e_fn = sp.lambdify(args_with_r, int_d2e, modules="numpy", cse=True)
    dle_fn = sp.lambdify(args_with_r, int_dle, modules="numpy", cse=True)

    pi_call = _make_trap_callable(integrand_fn)
    dpi_dl_call = _make_trap_callable(dl_fn)
    dpi_de_call = _make_trap_callable(de_fn)
    d2l_call = _make_trap_callable(d2l_fn)
    d2e_call = _make_trap_callable(d2e_fn)
    dle_call = _make_trap_callable(dle_fn)
    det_h_call = _make_det_h_callable(d2l_call, d2e_call, dle_call)

    return {
        "Pi": pi_call,
        "dPi_dlambda_0": dpi_dl_call,
        "dPi_deta_0": dpi_de_call,
        "det_H_analytical": det_h_call,
    }


def derive_symbolic_expressions(use_cache: bool = True) -> dict[str, Callable]:
    """Derive Pi, its first partials w.r.t. (lambda_0, eta_0), and det(H).

    Returns
    -------
    dict
        Mapping with keys ``Pi``, ``dPi_dlambda_0``, ``dPi_deta_0``,
        ``det_H_analytical``. Each value is a callable with signature
        ``f(alpha, alpha_p, p, lambda_0, eta_0)`` accepting NumPy scalars or
        arrays. Arrays are broadcast against each other; the trapezoidal sum
        is taken over the prescribed 1001-point uniform r-grid (PRD Section
        4.5) and returned with the broadcast output shape.

    Notes
    -----
    Results are cached to ``trained_models/symbolic_cache.pkl`` keyed by
    ``SYMBOLIC_VERSION``. Cache mismatches trigger a rebuild.
    """
    if use_cache and _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as fh:
                cached = cloudpickle.load(fh)
            if cached.get("version") == SYMBOLIC_VERSION:
                return cached["funcs"]
            logger.info("Symbolic cache version mismatch; rebuilding.")
        except (OSError, EOFError, cloudpickle.pickle.UnpicklingError) as exc:
            logger.warning("Failed to read symbolic cache (%s); rebuilding.", exc)

    funcs = _derive_all()
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "wb") as fh:
        cloudpickle.dump({"version": SYMBOLIC_VERSION, "funcs": funcs}, fh)
    return funcs
