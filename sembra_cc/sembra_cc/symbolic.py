"""Symbolic/numerical energy core for the charge-controlled (CC) scenario.

SymPy/NumPy implementation of the membrane total potential energy functional Pi
and its first and second derivatives with respect to the Ritz state variables.
The symbolic derivation runs once at import; the resulting integrands are
lambdified to NumPy and integrated by composite trapezoid over the radial
coordinate. This module is the reference implementation that ``torch_energy.py``
must agree with. The energy follows the v2 paper (Sections 2-4).
"""

import time

import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

from sembra_cc.calibrations import TRAPEZOIDAL_GRID_SIZE


def _build_symbolic_core():
    """Derive Pi's integrand, gradient, and Hessian and lambdify them.

    Returns:
        Tuple ``(pi_func, grad_funcs, hess_funcs)``: ``pi_func`` is the
        lambdified integrand pi_r, ``grad_funcs`` is a length-4 list of
        lambdified first-partial integrands (order lambda_0, a, eta_0, b), and
        ``hess_funcs`` is a 4x4 nested list of lambdified second-partial
        integrands. Every callable has signature
        ``(alpha, Phi, p, lambda_0, a, eta_0, b, r)`` and is vectorized over r.
    """
    alpha, Phi, p = sp.symbols("alpha Phi p", real=True)
    lambda_0, a, eta_0, b = sp.symbols("lambda_0 a eta_0 b", real=True)
    r = sp.symbols("r", real=True)

    state = (lambda_0, a, eta_0, b)

    # Ritz ansatz (paper Eq. 12) and stretches (paper Eq. 1).
    lam2 = r**2 + lambda_0 * (1 - r**2) + a * r**3 * (1 - r)
    eta = eta_0 * (1 - r**2) + b * r**3 * (1 - r)
    lam2p = sp.diff(lam2, r)
    etap = sp.diff(eta, r)
    lam1 = sp.sqrt((lam2 + r * lam2p) ** 2 + etap**2)
    lam3 = 1 / (lam1 * lam2)

    # Energy densities (paper Eqs. 2, 4, 7).
    U = (1 / alpha) * (lam1**alpha + lam2**alpha + lam3**alpha - 3)
    pressure = sp.Rational(1, 2) * p * (r * lam2) ** 2 * etap
    elec = +(Phi * lam3**2 / (2 * alpha)) * r  # CC electrostatic term (1/(2*alpha) factor).

    # Total potential energy integrand (paper Eq. 10). No leading 2*pi factor.
    pi_r = U * r + elec + pressure

    args = (alpha, Phi, p, lambda_0, a, eta_0, b, r)

    pi_func = sp.lambdify(args, pi_r, modules="numpy")
    grad_funcs = [
        sp.lambdify(args, sp.diff(pi_r, s), modules="numpy") for s in state
    ]
    hess_funcs = [
        [sp.lambdify(args, sp.diff(pi_r, si, sj), modules="numpy") for sj in state]
        for si in state
    ]
    return pi_func, grad_funcs, hess_funcs


_t0 = time.perf_counter()
_PI_FUNC, _GRAD_FUNCS, _HESS_FUNCS = _build_symbolic_core()
DERIVATION_SECONDS = time.perf_counter() - _t0

# Radial integration grid, built once.
R_GRID = np.linspace(0.0, 1.0, TRAPEZOIDAL_GRID_SIZE)


def _integrate(func, alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate a lambdified integrand on R_GRID and trapezoid-integrate it.

    Args:
        func: Lambdified integrand callable.
        alpha, Phi, p: Loading parameters.
        lambda_0, a, eta_0, b: Ritz state variables.

    Returns:
        float: The trapezoidal integral over r in [0, 1].
    """
    values = func(alpha, Phi, p, lambda_0, a, eta_0, b, R_GRID)
    values = np.broadcast_to(np.asarray(values, dtype=float), R_GRID.shape)
    return float(trapezoid(values, R_GRID))


def evaluate_pi(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate the total potential energy Pi at a state.

    Args:
        alpha, Phi, p: Loading parameters (Ogden exponent, electrical loading,
            inflation pressure).
        lambda_0, a, eta_0, b: Ritz state variables.

    Returns:
        float: Pi.
    """
    return _integrate(_PI_FUNC, alpha, Phi, p, lambda_0, a, eta_0, b)


def evaluate_gradient(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate the gradient of Pi w.r.t. the Ritz state variables.

    Args:
        alpha, Phi, p: Loading parameters.
        lambda_0, a, eta_0, b: Ritz state variables.

    Returns:
        np.ndarray: Shape (4,) = (dPi/dlambda_0, dPi/da, dPi/deta_0, dPi/db).
    """
    return np.array(
        [_integrate(g, alpha, Phi, p, lambda_0, a, eta_0, b) for g in _GRAD_FUNCS],
        dtype=float,
    )


def evaluate_hessian(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate the 4x4 Hessian of Pi w.r.t. the Ritz state variables.

    Args:
        alpha, Phi, p: Loading parameters.
        lambda_0, a, eta_0, b: Ritz state variables.

    Returns:
        np.ndarray: Shape (4, 4), symmetric by construction.
    """
    H = np.empty((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            H[i, j] = _integrate(
                _HESS_FUNCS[i][j], alpha, Phi, p, lambda_0, a, eta_0, b
            )
    return H


def evaluate_det_hessian(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate det(Hessian) of Pi at a state.

    Args:
        alpha, Phi, p: Loading parameters.
        lambda_0, a, eta_0, b: Ritz state variables.

    Returns:
        float: det(H) of the 4x4 Hessian.
    """
    return float(
        np.linalg.det(evaluate_hessian(alpha, Phi, p, lambda_0, a, eta_0, b))
    )
