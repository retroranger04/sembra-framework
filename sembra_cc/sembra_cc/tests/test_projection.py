"""Tests for the charge-controlled projection layer.

Uses verified equilibria from the training dataset (their residuals are ~1e-7,
see prompt 02c) as known points on the equilibrium manifold.
"""

import numpy as np
import torch

from sembra_cc.calibrations import PROJECTION_MAX_ITERATIONS, PROJECTION_TOLERANCE
from sembra_cc.data import load_full_dataset, split_stable_unstable
from sembra_cc.projection import ProjectionLayer

_STATE_COLS = ["lambda_0", "a", "eta_0", "b"]


def _mid_stable_rows(n=1):
    """Return ``n`` stable rows near the middle of the alpha range."""
    stable, _ = split_stable_unstable(load_full_dataset())
    alpha_mid = sorted(stable["alpha"].unique())[len(stable["alpha"].unique()) // 2]
    sub = stable[stable["alpha"] == alpha_mid].reset_index(drop=True)
    idx = np.linspace(len(sub) // 4, 3 * len(sub) // 4, n).astype(int)
    return sub.iloc[idx].reset_index(drop=True)


def _t(value):
    return torch.tensor(np.atleast_1d(np.asarray(value, dtype=float)), dtype=torch.float64)


def test_identity_on_equilibrium():
    """A state already on the manifold is returned unchanged in <=1 iteration."""
    row = _mid_stable_rows(1).iloc[0]
    proj = ProjectionLayer()
    a, eta_0, b, info = proj(
        _t(row["alpha"]), _t(row["Phi"]), _t(row["lambda_0"]),
        _t(row["a"]), _t(row["eta_0"]), _t(row["b"]),
    )
    a, eta_0, b = a.detach(), eta_0.detach(), b.detach()
    assert bool(info["converged"][0])
    assert int(info["iterations"][0]) <= 1
    assert float(info["final_residual_norm"][0]) < PROJECTION_TOLERANCE
    assert abs(float(a[0]) - row["a"]) < PROJECTION_TOLERANCE
    assert abs(float(eta_0[0]) - row["eta_0"]) < PROJECTION_TOLERANCE
    assert abs(float(b[0]) - row["b"]) < PROJECTION_TOLERANCE


def test_convergence_from_perturbation():
    """A 0.05 offset from equilibrium is projected back within the iteration cap."""
    row = _mid_stable_rows(1).iloc[0]
    proj = ProjectionLayer()
    a, eta_0, b, info = proj(
        _t(row["alpha"]), _t(row["Phi"]), _t(row["lambda_0"]),
        _t(row["a"] + 0.05), _t(row["eta_0"] + 0.05), _t(row["b"] + 0.05),
    )
    a, eta_0, b = a.detach(), eta_0.detach(), b.detach()
    assert bool(info["converged"][0])
    assert int(info["iterations"][0]) <= PROJECTION_MAX_ITERATIONS
    assert float(info["final_residual_norm"][0]) < PROJECTION_TOLERANCE
    # Recovered the original equilibrium.
    assert abs(float(a[0]) - row["a"]) < 1e-4
    assert abs(float(eta_0[0]) - row["eta_0"]) < 1e-4
    assert abs(float(b[0]) - row["b"]) < 1e-4


def test_differentiable_through_iteration():
    """Autograd flows from the projected outputs back to the initial guess."""
    row = _mid_stable_rows(1).iloc[0]
    proj = ProjectionLayer()
    a_init = _t(row["a"] + 0.02).requires_grad_(True)
    a, eta_0, b, _ = proj(
        _t(row["alpha"]), _t(row["Phi"]), _t(row["lambda_0"]),
        a_init, _t(row["eta_0"]), _t(row["b"]),
    )
    loss = (a**2 + eta_0**2 + b**2).sum()
    loss.backward()
    assert a_init.grad is not None
    grad = float(a_init.grad[0])
    assert np.isfinite(grad)
    assert grad != 0.0


def test_batch_shapes():
    """A batch of 4 configurations produces batch-shaped outputs."""
    rows = _mid_stable_rows(4)
    proj = ProjectionLayer()
    a, eta_0, b, info = proj(
        _t(rows["alpha"].to_numpy()), _t(rows["Phi"].to_numpy()),
        _t(rows["lambda_0"].to_numpy()), _t(rows["a"].to_numpy()),
        _t(rows["eta_0"].to_numpy()), _t(rows["b"].to_numpy()),
    )
    assert a.shape == (4,)
    assert eta_0.shape == (4,)
    assert b.shape == (4,)
    assert info["converged"].shape == (4,)
    assert info["iterations"].shape == (4,)
