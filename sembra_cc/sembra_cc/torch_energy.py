"""PyTorch energy core for the charge-controlled (CC) scenario.

PyTorch reimplementation of the membrane total potential energy functional Pi
and its gradient, used by the projection layer's autograd path. The integrand
mirrors the SymPy/NumPy reference in ``symbolic.py`` (v2 paper, Sections 2-4);
spatial r-derivatives are written in closed form for speed and exactness.
Gradients w.r.t. the state variables come from autograd through the trapezoidal
quadrature, which is exact because trapezoid integration is a linear
(weighted-sum) operator. Computation is batched and device-aware, running on
whatever device the input tensors live on.
"""

import torch

from sembra_cc.calibrations import TRAPEZOIDAL_GRID_SIZE


def _infer_device_dtype(values):
    """Infer the torch device and dtype to use from the first tensor input.

    Args:
        values: Iterable of inputs (Python floats or tensors).

    Returns:
        Tuple (device, dtype). Defaults to (None, torch.float64) when no
        floating-point tensor is present, so all-float inputs run in float64.
    """
    for v in values:
        if torch.is_tensor(v):
            dtype = v.dtype if v.is_floating_point() else torch.float64
            return v.device, dtype
    return None, torch.float64


def _as_column(x, device, dtype):
    """Coerce a scalar or shape-(B,) input to a column tensor of shape (B, 1).

    Args:
        x: Python float, 0-d tensor, or 1-d tensor of shape (B,).
        device: Target torch device.
        dtype: Target torch dtype.

    Returns:
        torch.Tensor of shape (B, 1).
    """
    t = torch.as_tensor(x, device=device, dtype=dtype)
    if t.dim() == 0:
        t = t.reshape(1)
    return t.reshape(-1, 1)


def _kinematics_torch(lambda_0, a, eta_0, b, r):
    """Compute the kinematic quantities entering the energy densities.

    Args:
        lambda_0, a, eta_0, b: Column tensors of shape (B, 1).
        r: Radial grid row tensor of shape (1, N).

    Returns:
        Tuple (lam1, lam2, lam3, etap) of tensors broadcasting to (B, N).
    """
    lam2 = r**2 + lambda_0 * (1 - r**2) + a * r**3 * (1 - r)
    # Closed-form spatial r-derivatives of the Ritz ansatz.
    lam2p = 2 * r * (1 - lambda_0) + a * (3 * r**2 - 4 * r**3)
    etap = -2 * eta_0 * r + 3 * b * r**2 - 4 * b * r**3
    # Meridional stretch as a Euclidean length ratio (paper Eq. 1).
    lam1 = torch.sqrt((lam2 + r * lam2p) ** 2 + etap**2)
    lam3 = 1.0 / (lam1 * lam2)
    return lam1, lam2, lam3, etap


def pi_integrand_torch(alpha, Phi, p, lambda_0, a, eta_0, b, r_grid):
    """Evaluate the differentiable energy integrand pi_r over a radial grid.

    Args:
        alpha, Phi, p, lambda_0, a, eta_0, b: Python floats, 0-d tensors, or
            shape-(B,) tensors of loading and state parameters.
        r_grid: 1-d tensor of radial coordinates, shape (N,).

    Returns:
        torch.Tensor of shape (B, N), differentiable w.r.t. tensor inputs.
    """
    device, dtype = _infer_device_dtype(
        (alpha, Phi, p, lambda_0, a, eta_0, b, r_grid)
    )
    alpha, Phi, p, lambda_0, a, eta_0, b = (
        _as_column(v, device, dtype)
        for v in (alpha, Phi, p, lambda_0, a, eta_0, b)
    )
    r = torch.as_tensor(r_grid, device=device, dtype=dtype).reshape(1, -1)

    lam1, lam2, lam3, etap = _kinematics_torch(lambda_0, a, eta_0, b, r)

    U = (1 / alpha) * (lam1**alpha + lam2**alpha + lam3**alpha - 3)
    pressure = 0.5 * p * (r * lam2) ** 2 * etap
    elec = +(Phi * lam3**2 / (2 * alpha)) * r  # CC electrostatic term (1/(2*alpha) factor).
    return U * r + elec + pressure


def pi_total_torch(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate the total potential energy Pi for a batch.

    Args:
        alpha, Phi, p, lambda_0, a, eta_0, b: Python floats, 0-d tensors, or
            shape-(B,) tensors of loading and state parameters.

    Returns:
        torch.Tensor of shape (B,): the integral trapz(pi_r, r).
    """
    device, dtype = _infer_device_dtype((alpha, Phi, p, lambda_0, a, eta_0, b))
    r_grid = torch.linspace(
        0.0, 1.0, TRAPEZOIDAL_GRID_SIZE, device=device, dtype=dtype
    )
    integrand = pi_integrand_torch(alpha, Phi, p, lambda_0, a, eta_0, b, r_grid)
    return torch.trapezoid(integrand, r_grid, dim=-1)


def pi_gradient_torch(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Evaluate the gradient of Pi w.r.t. the Ritz state variables via autograd.

    Args:
        alpha, Phi, p, lambda_0, a, eta_0, b: Python floats, 0-d tensors, or
            shape-(B,) tensors of loading and state parameters.

    Returns:
        torch.Tensor of shape (B, 4): (dPi/dlambda_0, dPi/da, dPi/deta_0,
        dPi/db). Differentiable (create_graph=True) for use inside the
        projection layer's autograd path.
    """
    device, dtype = _infer_device_dtype((alpha, Phi, p, lambda_0, a, eta_0, b))
    leaves = [
        _as_column(v, device, dtype).reshape(-1).detach().clone().requires_grad_(True)
        for v in (lambda_0, a, eta_0, b)
    ]
    total = pi_total_torch(alpha, Phi, p, *leaves).sum()
    grads = torch.autograd.grad(total, leaves, create_graph=True)
    return torch.stack(grads, dim=-1)
