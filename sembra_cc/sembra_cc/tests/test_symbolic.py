"""Tests for the charge-controlled symbolic/NumPy and PyTorch energy cores.

Verifies integrand-level symbolic differentiation against finite differences,
symbolic/PyTorch agreement on Pi and its gradient, Hessian symmetry, and the
integration grid-size calibration.
"""

import math

import numpy as np
import torch

from sembra_cc.calibrations import TRAPEZOIDAL_GRID_SIZE
from sembra_cc.symbolic import evaluate_gradient, evaluate_hessian, evaluate_pi
from sembra_cc.torch_energy import pi_gradient_torch, pi_total_torch

# CC test point: (alpha, Phi, p, lambda_0, a, eta_0, b).
ALPHA, PHI, P = 1.6, 0.1, 0.5
LAMBDA_0, A, ETA_0, B = 0.65, 0.04, 0.45, -0.06


def _torch64(value):
    """Build a float64 CPU tensor from a Python scalar."""
    return torch.tensor(value, dtype=torch.float64)


def test_integrand_level_differentiation():
    """dPi/dlambda_0 from the symbolic gradient matches central finite diff."""
    analytic = evaluate_gradient(ALPHA, PHI, P, LAMBDA_0, A, ETA_0, B)[0]
    h = 1e-5
    fd = (
        evaluate_pi(ALPHA, PHI, P, LAMBDA_0 + h, A, ETA_0, B)
        - evaluate_pi(ALPHA, PHI, P, LAMBDA_0 - h, A, ETA_0, B)
    ) / (2 * h)
    assert abs(analytic - fd) < 1e-5


def test_symbolic_vs_torch_total():
    """Symbolic Pi agrees with the PyTorch total to rel_tol 1e-8."""
    sym = evaluate_pi(ALPHA, PHI, P, LAMBDA_0, A, ETA_0, B)
    tor = float(
        pi_total_torch(
            ALPHA, PHI, P,
            _torch64(LAMBDA_0), _torch64(A), _torch64(ETA_0), _torch64(B),
        ).squeeze()
    )
    assert math.isclose(sym, tor, rel_tol=1e-8, abs_tol=0.0)


def test_symbolic_vs_torch_gradient():
    """Symbolic gradient agrees with the PyTorch autograd gradient."""
    sym = evaluate_gradient(ALPHA, PHI, P, LAMBDA_0, A, ETA_0, B)
    tor = (
        pi_gradient_torch(
            ALPHA, PHI, P,
            _torch64(LAMBDA_0), _torch64(A), _torch64(ETA_0), _torch64(B),
        )
        .squeeze(0)
        .detach()
        .numpy()
    )
    assert np.allclose(sym, tor, rtol=1e-7, atol=1e-12)


def test_hessian_symmetry():
    """The symbolic Hessian is symmetric."""
    H = evaluate_hessian(ALPHA, PHI, P, LAMBDA_0, A, ETA_0, B)
    assert np.max(np.abs(H - H.T)) < 1e-10


def test_grid_size_sanity():
    """The trapezoidal grid-size calibration is 2001."""
    assert TRAPEZOIDAL_GRID_SIZE == 2001
