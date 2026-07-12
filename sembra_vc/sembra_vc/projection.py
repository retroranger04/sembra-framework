"""Equilibrium projection layer for the voltage-controlled scenario.

A non-trainable PyTorch module that refines a candidate Ritz sub-state
``(a, eta_0, b)`` so that the three-equation equilibrium sub-system

    dPi/da = dPi/deta_0 = dPi/db = 0

is satisfied at the fixed loading ``(alpha, Phi)`` and fixed conditioning
``lambda_0``.

``p`` is not a free input to this layer. As the PRD specifies, inside the
projection ``p`` is a *dependent* quantity fixed by the fourth equilibrium
equation ``dPi/dlambda_0 = 0`` (which the upstream 1D inversion also enforces).
Because Pi is linear in ``p`` -- ``Pi(p) = Pi_0 + p * Pi_press`` -- that fourth
equation is linear in ``p`` and solved in closed form at each iterate:

    p_dep = - (dPi_0/dlambda_0) / (dPi_press/dlambda_0)

and the residual is evaluated at that ``p_dep``. ``Pi_0`` and ``Pi_press`` are
recovered from two energy evaluations at ``p = 0`` and ``p = 1``.

The solver is regularized Gauss-Newton. Every operation is a standard,
differentiable PyTorch op, so autograd flows through the whole iteration: the
network's training loss (computed on the projected state) back-propagates through
the projection. Energy gradients come from ``torch_energy.pi_total_torch`` via
autograd (not ``pi_gradient_torch``, which detaches its inputs and would sever
the graph).
"""

import torch
from torch import nn

from sembra_vc.calibrations import (
    PROJECTION_MAX_ITERATIONS,
    PROJECTION_REGULARIZATION,
    PROJECTION_TOLERANCE,
)
from sembra_vc.torch_energy import pi_total_torch


def _state_gradient(alpha, Phi, p, lambda_0, a, eta_0, b):
    """Gradient of Pi w.r.t. (lambda_0, a, eta_0, b), shape ``(B, 4)``.

    Built with ``create_graph=True`` so it can be differentiated again (for the
    Jacobian) and so the outer training gradient flows through it. The four state
    tensors must already require grad.
    """
    total = pi_total_torch(alpha, Phi, p, lambda_0, a, eta_0, b).sum()
    grads = torch.autograd.grad(total, (lambda_0, a, eta_0, b), create_graph=True)
    return torch.stack(grads, dim=-1)


def _residual(alpha, Phi, lambda_0, a, eta_0, b):
    """Return the pressure-eliminated 3-vector residual ``F`` and ``p_dep``.

    ``F = (dPi/da, dPi/deta_0, dPi/db)`` evaluated at the ``p`` that satisfies
    ``dPi/dlambda_0 = 0``. Shapes: ``F`` is ``(B, 3)``, ``p_dep`` is ``(B,)``.
    """
    g0 = _state_gradient(alpha, Phi, 0.0, lambda_0, a, eta_0, b)  # Pi_0 gradient
    g1 = _state_gradient(alpha, Phi, 1.0, lambda_0, a, eta_0, b)  # Pi_0 + Pi_press
    g_press = g1 - g0  # gradient of the pressure functional Pi_press
    p_dep = -g0[:, 0] / g_press[:, 0]  # solves dPi/dlambda_0 = 0 (linear in p)
    F = g0[:, 1:] + p_dep.unsqueeze(-1) * g_press[:, 1:]
    return F, p_dep


def _jacobian(residual, a, eta_0, b):
    """3x3 batched Jacobian ``J[:, i, j] = dF_i / d(a, eta_0, b)_j``, ``(B, 3, 3)``."""
    rows = []
    for i in range(3):
        grads = torch.autograd.grad(
            residual[:, i].sum(), (a, eta_0, b), create_graph=True, retain_graph=True
        )
        rows.append(torch.stack(grads, dim=-1))
    return torch.stack(rows, dim=1)


def _require_grad(x):
    """Return a shape-(B,) tensor view of ``x`` that requires grad.

    Preserves the outer autograd graph when ``x`` already requires grad (training
    path); otherwise starts a fresh grad-enabled leaf (inference path).
    """
    x = torch.atleast_1d(x)
    if x.requires_grad:
        return x
    return x.detach().clone().requires_grad_(True)


class ProjectionLayer(nn.Module):
    """Gauss-Newton projection onto the equilibrium manifold for (a, eta_0, b).

    Operates at fixed ``(alpha, Phi, lambda_0)``. Tolerance, iteration cap, and
    Tikhonov regularization all come from ``calibrations``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tolerance = PROJECTION_TOLERANCE
        self.max_iterations = PROJECTION_MAX_ITERATIONS
        self.regularization = PROJECTION_REGULARIZATION

    def forward(
        self,
        alpha: torch.Tensor,
        Phi: torch.Tensor,
        lambda_0: torch.Tensor,
        a_init: torch.Tensor,
        eta_0_init: torch.Tensor,
        b_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Project ``(a_init, eta_0_init, b_init)`` onto the equilibrium manifold.

        All state inputs are batch tensors of shape ``(B,)`` (0-d tensors are
        treated as a batch of one). Returns ``(a_proj, eta_0_proj, b_proj, info)``
        with ``info`` holding ``converged`` (bool), ``iterations`` (long), and
        ``final_residual_norm`` (float), each shaped ``(B,)``.
        """
        with torch.enable_grad():
            # lambda_0 is fixed but must carry grad for the dPi/dlambda_0 term.
            lambda_0 = _require_grad(lambda_0)
            a = _require_grad(a_init)
            eta_0 = _require_grad(eta_0_init)
            b = _require_grad(b_init)

            batch = a.shape[0]
            device = a.device
            converged = torch.zeros(batch, dtype=torch.bool, device=device)
            iterations = torch.zeros(batch, dtype=torch.long, device=device)
            eye = self.regularization * torch.eye(3, dtype=a.dtype, device=device)

            for _ in range(self.max_iterations):
                F, _ = _residual(alpha, Phi, lambda_0, a, eta_0, b)  # (B, 3)
                fnorm = torch.linalg.norm(F, dim=-1)  # (B,)
                converged = converged | (fnorm < self.tolerance)
                active = ~converged
                if not bool(active.any()):
                    break

                J = _jacobian(F, a, eta_0, b)  # (B, 3, 3)
                Jt = J.transpose(-1, -2)
                lhs = Jt @ J + eye
                rhs = -(Jt @ F.unsqueeze(-1))
                delta = torch.linalg.solve(lhs, rhs).squeeze(-1)  # (B, 3)

                # Step only the samples that have not converged yet.
                step = torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))
                a = a + step[:, 0]
                eta_0 = eta_0 + step[:, 1]
                b = b + step[:, 2]
                iterations = iterations + active.long()

            # Residual at the returned state (captures late/failed convergence).
            final_F, _ = _residual(alpha, Phi, lambda_0, a, eta_0, b)
            final_norm = torch.linalg.norm(final_F, dim=-1).detach()
            converged = converged | (final_norm < self.tolerance)

        info = {
            "converged": converged,
            "iterations": iterations,
            "final_residual_norm": final_norm,
        }
        return a, eta_0, b, info
