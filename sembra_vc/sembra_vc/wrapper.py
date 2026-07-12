"""User-facing prediction wrapper for the voltage-controlled scenario.

Exposes :class:`PredictionResult` and :func:`predict`, the single entry point a
user calls. ``predict`` validates the loading parameters and routes the query
through one of three cases based on ``delta = (p - p_crit) / p_crit`` (plus the
trivial-p and no-stable-region short-circuits):

- **stable** (``delta < -GREY_ZONE_THRESHOLD``): NN forward -> projection layer.
- **near_fold** (``|delta| <= GREY_ZONE_THRESHOLD``): NN warm-start -> 4D symbolic
  Newton at the user's p; a converged equilibrium near the limit point.
- **no_equilibrium** (``delta > GREY_ZONE_THRESHOLD``): p exceeds the limit-point
  pressure, which is the maximum sustainable pressure in this ansatz, so no
  static equilibrium exists at the requested p. Rather than extrapolate the NN,
  the wrapper reports the real converged equilibrium at the snap-through boundary
  ``(alpha, Phi, p_crit)`` and flags ``requested_p_exceeds_p_crit``.

This three-case architecture is the authoritative router per DEV-005 (it
supersedes the PRD router until PRD is updated).

Resources (GP envelopes, network) are loaded lazily and cached. The network is
only needed on the stable branch, so every other branch works before weights
exist.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root

from sembra_vc.calibrations import (
    ALPHA_MAX,
    ALPHA_MIN,
    GREY_ZONE_THRESHOLD,
    INVERSION_LAMBDA_MIN,
    INVERSION_LIMIT_MARGIN,
    PAST_LIMIT_A_INIT,
    PAST_LIMIT_B_INIT,
    PAST_LIMIT_ETA0_INIT,
    PAST_LIMIT_MARGIN_MAX,
    PAST_LIMIT_MARGIN_MIN,
    PHI_MAX,
    PHI_MIN,
    STRESS_DISPLAY_GRID_SIZE,
    TRIVIAL_P_THRESHOLD,
)
from sembra_vc.envelope import load_envelopes, predict_lambda_limit, predict_p_crit
from sembra_vc.symbolic import evaluate_det_hessian, evaluate_gradient
from sembra_vc.utils import (
    compute_max_electric_field,
    compute_stress,
    detect_wrinkling,
)


@dataclass
class PredictionResult:
    alpha: float
    Phi: float
    p: float
    lambda_0: float
    a: float
    eta_0: float
    b: float
    det_H: float
    stable: bool
    message: str
    diagnostics: dict


# ---------------------------------------------------------------------------
# Resource loading (lazy, cached)
# ---------------------------------------------------------------------------
_ENVELOPES = None
_NETWORK = None


def _get_envelopes():
    global _ENVELOPES
    if _ENVELOPES is None:
        _ENVELOPES = load_envelopes()
    return _ENVELOPES


def _get_network():
    """Lazily load the trained network (stable branch only)."""
    global _NETWORK
    if _NETWORK is None:
        from sembra_vc.network import load_network

        _NETWORK = load_network()
    return _NETWORK


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(alpha, Phi, p) -> None:
    for name, value in (("alpha", alpha), ("Phi", Phi), ("p", p)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a real number; got {value!r}.")
    if not (ALPHA_MIN <= alpha <= ALPHA_MAX):
        raise ValueError(f"alpha must be in [{ALPHA_MIN}, {ALPHA_MAX}]; got {alpha}.")
    if not (PHI_MIN <= Phi <= PHI_MAX):
        raise ValueError(f"Phi must be in [{PHI_MIN}, {PHI_MAX}]; got {Phi}.")
    if p < 0.0:
        raise ValueError(f"p must be greater than or equal to zero; got {p}.")


# ---------------------------------------------------------------------------
# Diagnostics assembly
# ---------------------------------------------------------------------------
def _diagnostics(
    p_crit,
    lambda_limit,
    branch,
    equilibrium_residual_norm,
    projection_residual_norm=None,
    projection_converged=True,
    projection_iterations=0,
    newton_converged=None,
    newton_iterations=None,
    equilibrium_exists=True,
    near_fold_warning=False,
    requested_p_exceeds_p_crit=False,
    boundary_margin=None,
):
    """Assemble the diagnostics dict with a consistent set of keys.

    ``residual_norm`` (the PRD-required key) is the full 4D equilibrium residual;
    ``equilibrium_residual_norm`` is the same value under an unambiguous name, and
    ``projection_residual_norm`` is the projection layer's inner 3D residual
    (populated on the stable branch only). The failure-mode keys (``det_H``,
    ``t_1``, ``t_2``, ``E_max``, ``wrinkling_detected``) are merged in by each
    branch handler so the dict has identical keys across cases.

    ``boundary_margin`` is ``None`` unless the near_fold / no_equilibrium adaptive
    solve had to retreat below the fold to converge, in which case it is the
    fractional retreat margin actually used (see DEV-018).
    """
    return {
        "p_crit": p_crit,
        "lambda_limit": lambda_limit,
        "branch": branch,
        "residual_norm": equilibrium_residual_norm,
        "equilibrium_residual_norm": equilibrium_residual_norm,
        "projection_residual_norm": projection_residual_norm,
        "projection_converged": projection_converged,
        "projection_iterations": projection_iterations,
        "newton_converged": newton_converged,
        "newton_iterations": newton_iterations,
        "equilibrium_exists": equilibrium_exists,
        "near_fold_warning": near_fold_warning,
        "requested_p_exceeds_p_crit": requested_p_exceeds_p_crit,
        "boundary_margin": boundary_margin,
    }


def _failure_mode_diagnostics(alpha, Phi, p_eval, state) -> dict:
    """Failure-mode diagnostics (det H, stresses, E_max, wrinkling) at ``state``.

    Evaluated at ``p_eval`` (the pressure at which ``state`` is an equilibrium).
    """
    r_grid = np.linspace(0.0, 1.0, STRESS_DISPLAY_GRID_SIZE)
    stress = compute_stress(alpha, Phi, p_eval, state, r_grid)
    wrinkling = detect_wrinkling(stress, r_grid)
    return {
        "det_H": evaluate_det_hessian(alpha, Phi, p_eval, *state),
        "t_1": stress["t_1"],
        "t_2": stress["t_2"],
        "E_max": compute_max_electric_field(state),
        "wrinkling_detected": wrinkling["detected"],
    }


def _empty_failure_mode_diagnostics() -> dict:
    """Failure-mode keys with null values, for branches without a valid state."""
    return {
        "det_H": math.nan,
        "t_1": None,
        "t_2": None,
        "E_max": math.nan,
        "wrinkling_detected": None,
    }


# ---------------------------------------------------------------------------
# Equilibrium solves
# ---------------------------------------------------------------------------
def _residual_norm_4d(alpha, Phi, p, state) -> float:
    """L2 norm of the full 4D equilibrium residual at ``state``."""
    return float(np.linalg.norm(evaluate_gradient(alpha, Phi, p, *state)))


def _solve_equilibrium_4d(alpha, Phi, p, lambda_limit):
    """Solve grad(Pi) = 0 for (lambda_0, a, eta_0, b) at fixed (alpha, Phi, p).

    Warm-started from the GP limit-point stretch and the calibrated neutral
    guesses. Returns ``(state, converged, iterations)``.
    """
    x0 = [lambda_limit, PAST_LIMIT_A_INIT, PAST_LIMIT_ETA0_INIT, PAST_LIMIT_B_INIT]
    solution = root(
        lambda x: evaluate_gradient(alpha, Phi, p, x[0], x[1], x[2], x[3]),
        x0,
        method="hybr",
    )
    state = tuple(float(v) for v in solution.x)
    return state, bool(solution.success), int(solution.nfev)


def _solve_equilibrium_4d_adaptive(alpha, Phi, target_p, lambda_limit):
    """Solve grad(Pi) = 0 at ``target_p``, retreating below it if the fold is singular.

    Near the fold ``p_crit`` the equilibrium Jacobian is singular (det H = 0), so a
    solve exactly at ``target_p`` frequently fails. This first tries ``target_p``;
    on failure it retreats to ``target_p * (1 - margin)`` over a margin ladder from
    ``PAST_LIMIT_MARGIN_MIN`` to ``PAST_LIMIT_MARGIN_MAX`` until the solve converges.

    Returns ``(state, converged, iterations, p_solved, margin_used)``. ``margin_used``
    is ``0.0`` when convergence was achieved exactly at ``target_p`` (no retreat).
    """
    state, converged, iterations = _solve_equilibrium_4d(
        alpha, Phi, target_p, lambda_limit
    )
    if converged:
        return state, True, iterations, target_p, 0.0

    p_solved = target_p
    margin = PAST_LIMIT_MARGIN_MAX
    for margin in np.linspace(PAST_LIMIT_MARGIN_MIN, PAST_LIMIT_MARGIN_MAX, 6):
        p_solved = target_p * (1.0 - float(margin))
        state, converged, iterations = _solve_equilibrium_4d(
            alpha, Phi, p_solved, lambda_limit
        )
        if converged:
            return state, True, iterations, p_solved, float(margin)
    return state, False, iterations, p_solved, float(margin)


def _invert_p_to_lambda0(network, alpha, Phi, p, lambda_limit):
    """Recover lambda_0 from p by inverting the network's p(lambda_0) curve."""
    import torch

    upper = lambda_limit - INVERSION_LIMIT_MARGIN
    device = next(network.parameters()).device  # move inputs to the network's device

    def p_gap(lambda_0):
        x = torch.tensor([[alpha, Phi, lambda_0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            p_pred = float(network(x)[0, 0])
        return p_pred - p

    return brentq(p_gap, INVERSION_LAMBDA_MIN, upper)


# ---------------------------------------------------------------------------
# Branch handlers
# ---------------------------------------------------------------------------
def _trivial_result(alpha, Phi, p) -> PredictionResult:
    state = (1.0, 0.0, 0.0, 0.0)
    det_H = evaluate_det_hessian(alpha, Phi, p, *state)
    residual = _residual_norm_4d(alpha, Phi, p, state)
    envelopes = _get_envelopes()
    p_crit = predict_p_crit(envelopes[0], alpha, Phi)
    lambda_limit = predict_lambda_limit(envelopes[1], alpha, Phi)
    diagnostics = _diagnostics(p_crit, lambda_limit, "stable", residual)
    diagnostics.update(_failure_mode_diagnostics(alpha, Phi, p, state))
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=1.0, a=0.0, eta_0=0.0, b=0.0,
        det_H=det_H, stable=True,
        message="Trivial case: pressure below threshold; membrane is undeformed.",
        diagnostics=diagnostics,
    )


def _no_stable_region_result(alpha, Phi, p, p_crit, lambda_limit) -> PredictionResult:
    diagnostics = _diagnostics(
        p_crit, lambda_limit, "no_stable_region", math.nan,
        equilibrium_exists=False,
    )
    diagnostics.update(_empty_failure_mode_diagnostics())
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=math.nan, a=math.nan, eta_0=math.nan, b=math.nan,
        det_H=math.nan, stable=False,
        message=(
            "No stable region: the GP envelope gives a non-positive critical "
            "pressure for this (alpha, Phi); no stable equilibrium exists."
        ),
        diagnostics=diagnostics,
    )


def _near_fold_result(alpha, Phi, p, p_crit, lambda_limit) -> PredictionResult:
    """Near-fold handler: the Newton-converged 4D equilibrium near the limit point.

    ``stable`` reflects the branch the query fell into (True), not whether the
    numerical solve succeeded; a solve failure is signalled by
    ``equilibrium_exists`` / ``newton_converged`` instead.
    """
    state, converged, iterations, p_solved, margin = _solve_equilibrium_4d_adaptive(
        alpha, Phi, p, lambda_limit
    )

    if converged:
        det_H = evaluate_det_hessian(alpha, Phi, p_solved, *state)
        residual = _residual_norm_4d(alpha, Phi, p_solved, state)
        failure_modes = _failure_mode_diagnostics(alpha, Phi, p_solved, state)
    else:
        # Explicit, non-silent fallback: neutral state flagged as non-converged.
        state = (lambda_limit, 0.0, 0.0, 0.0)
        det_H = math.nan
        residual = math.nan
        failure_modes = _empty_failure_mode_diagnostics()

    # Report the retreat margin only when the adaptive solve actually retreated.
    boundary_margin = margin if (converged and margin > 0.0) else None

    message = (
        "Near fold: pressure is within +/-0.5% of the critical pressure. The "
        "reported state is the Newton-converged equilibrium near the limit point; "
        "treat stability as marginal (near-fold warning set)."
    )
    if boundary_margin is not None:
        message += (
            f" Diagnostics were computed at {boundary_margin * 100:.1f}% below the "
            "requested pressure for numerical convergence near the singular fold."
        )
    if not converged:
        message += (
            " WARNING: the 4D Newton solve did not converge even after retreating "
            "below the fold; the reported state is a non-converged fallback and "
            "should not be trusted."
        )

    diagnostics = _diagnostics(
        p_crit, lambda_limit, "near_fold", residual,
        newton_converged=converged, newton_iterations=iterations,
        equilibrium_exists=converged, near_fold_warning=True,
        boundary_margin=boundary_margin,
    )
    diagnostics.update(failure_modes)
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=state[0], a=state[1], eta_0=state[2], b=state[3],
        det_H=det_H, stable=True,
        message=message,
        diagnostics=diagnostics,
    )


def _no_equilibrium_result(alpha, Phi, p, p_crit, lambda_limit) -> PredictionResult:
    """No-equilibrium handler: p exceeds the fold's pressure maximum.

    No static equilibrium exists at the requested p. Rather than extrapolate the
    network outside its training distribution, report the real converged
    equilibrium at the snap-through boundary ``(alpha, Phi, p_crit)`` and flag
    that the request exceeded p_crit.
    """
    state, converged, iterations, p_solved, margin = _solve_equilibrium_4d_adaptive(
        alpha, Phi, p_crit, lambda_limit
    )

    if converged:
        det_H = evaluate_det_hessian(alpha, Phi, p_solved, *state)
        residual = _residual_norm_4d(alpha, Phi, p_solved, state)
        failure_modes = _failure_mode_diagnostics(alpha, Phi, p_solved, state)
    else:
        state = (lambda_limit, 0.0, 0.0, 0.0)
        det_H = math.nan
        residual = math.nan
        failure_modes = _empty_failure_mode_diagnostics()

    # Report the retreat margin only when the adaptive solve actually retreated
    # below p_crit (the fold Jacobian is singular exactly at p_crit).
    boundary_margin = margin if (converged and margin > 0.0) else None

    message = (
        f"Requested p={p:.4f} exceeds p_crit={p_crit:.4f} at (α={alpha}, Φ={Phi}).\n"
        f"Diagnostics shown are at the snap-through threshold "
        f"(λ_limit={lambda_limit:.4f}),\n"
        f"not at requested p."
    )
    if boundary_margin is not None:
        message += (
            f"\nThe boundary state was computed at {boundary_margin * 100:.1f}% below "
            "p_crit for numerical convergence near the singular fold."
        )
    if not converged:
        message += (
            "\nWARNING: the 4D Newton solve did not converge even after retreating "
            "below p_crit; diagnostics are unavailable for this query."
        )

    diagnostics = _diagnostics(
        p_crit, lambda_limit, "no_equilibrium", residual,
        newton_converged=converged, newton_iterations=iterations,
        equilibrium_exists=False, requested_p_exceeds_p_crit=True,
        boundary_margin=boundary_margin,
    )
    diagnostics.update(failure_modes)
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=state[0], a=state[1], eta_0=state[2], b=state[3],
        det_H=det_H, stable=False,
        message=message,
        diagnostics=diagnostics,
    )


def _stable_branch_result(alpha, Phi, p, p_crit, lambda_limit) -> PredictionResult:
    """Stable branch: network initial guess -> projection -> diagnostics.

    Requires trained network weights (available from prompt 05).
    """
    import torch

    from sembra_vc.projection import ProjectionLayer

    network = _get_network()
    device = next(network.parameters()).device  # move inputs to the network's device
    lambda_0 = _invert_p_to_lambda0(network, alpha, Phi, p, lambda_limit)

    x = torch.tensor([[alpha, Phi, lambda_0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        _p_pred, a_hat, eta_hat, b_hat = (float(v) for v in network(x)[0])

    proj = ProjectionLayer()

    def _t(value):
        return torch.tensor([value], dtype=torch.float64)

    a_t, eta_t, b_t, info = proj(
        _t(alpha), _t(Phi), _t(lambda_0), _t(a_hat), _t(eta_hat), _t(b_hat)
    )
    state = (lambda_0, float(a_t[0]), float(eta_t[0]), float(b_t[0]))

    det_H = evaluate_det_hessian(alpha, Phi, p, *state)
    residual = _residual_norm_4d(alpha, Phi, p, state)
    diagnostics = _diagnostics(
        p_crit, lambda_limit, "stable", residual,
        projection_residual_norm=float(info["final_residual_norm"][0]),
        projection_converged=bool(info["converged"][0]),
        projection_iterations=int(info["iterations"][0]),
    )
    diagnostics.update(_failure_mode_diagnostics(alpha, Phi, p, state))
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=state[0], a=state[1], eta_0=state[2], b=state[3],
        det_H=det_H, stable=bool(det_H > 0),
        message="Stable branch: equilibrium refined onto the manifold by projection.",
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def predict(alpha: float, Phi: float, p: float) -> PredictionResult:
    """Predict the equilibrium state and stability at ``(alpha, Phi, p)``.

    Returns a :class:`PredictionResult`. Raises ``ValueError`` for out-of-range or
    non-numeric inputs.
    """
    _validate(alpha, Phi, p)
    alpha, Phi, p = float(alpha), float(Phi), float(p)

    # Trivial shortcut: pressure at or below the numerical-zero threshold.
    if p < TRIVIAL_P_THRESHOLD:
        return _trivial_result(alpha, Phi, p)

    envelopes = _get_envelopes()
    p_crit = predict_p_crit(envelopes[0], alpha, Phi)
    lambda_limit = predict_lambda_limit(envelopes[1], alpha, Phi)

    if p_crit <= 0.0:
        return _no_stable_region_result(alpha, Phi, p, p_crit, lambda_limit)

    delta = (p - p_crit) / p_crit
    if delta < -GREY_ZONE_THRESHOLD:
        return _stable_branch_result(alpha, Phi, p, p_crit, lambda_limit)
    if delta <= GREY_ZONE_THRESHOLD:
        return _near_fold_result(alpha, Phi, p, p_crit, lambda_limit)
    return _no_equilibrium_result(alpha, Phi, p, p_crit, lambda_limit)
