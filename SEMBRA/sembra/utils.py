"""Notebook post-processing utilities: deformed-shape plot, stress evaluation, envelope contour."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sembra.envelope import load_envelope_models, predict_p_crit
from sembra.wrapper import predict


def _ensure_axes(ax: Any) -> Any:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    return ax


def _ritz_stretches(lambda_0: float, eta_0: float, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    one_minus_r2 = 1.0 - r * r
    lambda_2 = r * r + lambda_0 * one_minus_r2
    dlambda2_dr = 2.0 * r * (1.0 - lambda_0)
    deta_dr = -2.0 * eta_0 * r
    inner = lambda_2 + r * dlambda2_dr
    lambda_1 = np.sqrt(inner * inner + deta_dr * deta_dr)
    lambda_3 = 1.0 / (lambda_1 * lambda_2)
    return lambda_1, lambda_2, lambda_3


def plot_shape(alpha: float, alpha_p: float, p: float, ax: Any | None = None) -> Any:
    """Plot the deformed shape ``eta(r)`` from the predicted equilibrium (Section 4.2).

    Under DEC-012 the wrapper always returns populated state for inputs that
    pass validation, including grey-zone and past-limit cases, so this
    function plots the membrane equally well in stable and unstable regimes.

    Raises
    ------
    ValueError
        Only if the wrapper returns ``NaN`` for ``lambda_0`` or ``eta_0`` —
        i.e. when the envelope predicts no equilibrium (``p_crit ≤ 0``).
    """
    result = predict(alpha, alpha_p, p)
    if math.isnan(result.eta_0) or math.isnan(result.lambda_0):
        raise ValueError(
            "Cannot plot shape: wrapper returned NaN state "
            f"for (alpha={alpha}, alpha_p={alpha_p}, p={p}). Message: {result.message}"
        )
    r = np.linspace(0.0, 1.0, 1001)
    eta = result.eta_0 * (1.0 - r * r)
    ax = _ensure_axes(ax)
    ax.plot(r, eta)
    ax.set_xlabel("r")
    ax.set_ylabel("eta(r)")
    stability_label = "stable" if result.stable else "UNSTABLE (past limit)"
    ax.set_title(
        f"Deformed shape (alpha={alpha}, alpha_p={alpha_p}, p={p}) — {stability_label}\n"
        f"lambda_0={result.lambda_0:.4f}, eta_0={result.eta_0:.4f}"
    )
    return ax


def compute_stress(alpha: float, alpha_p: float, p: float, r: np.ndarray) -> dict:
    """Return the principal Cauchy stresses at the predicted equilibrium.

    Formulas per DEC-008 (verified):
        sigma_1(r) = lambda_1(r)^alpha - lambda_3(r)^alpha - alpha_p / lambda_3(r)^2
        sigma_2(r) = lambda_2(r)^alpha - lambda_3(r)^alpha - alpha_p / lambda_3(r)^2

    Tensile-positive sign convention. Returns a dict with keys ``r``,
    ``sigma_1``, ``sigma_2``, ``lambda_0``, ``eta_0``.

    Under DEC-012 the wrapper always returns populated state, so this
    function evaluates stresses for stable, grey-zone, and past-limit
    equilibria alike.

    Raises
    ------
    ValueError
        Only if the wrapper returns ``NaN`` for ``lambda_0`` or ``eta_0`` —
        i.e. when the envelope predicts no equilibrium (``p_crit ≤ 0``).
    """
    result = predict(alpha, alpha_p, p)
    if math.isnan(result.eta_0) or math.isnan(result.lambda_0):
        raise ValueError(
            "Cannot compute stress: wrapper returned NaN state "
            f"for (alpha={alpha}, alpha_p={alpha_p}, p={p}). Message: {result.message}"
        )
    r_arr = np.asarray(r, dtype=float)
    lambda_1, lambda_2, lambda_3 = _ritz_stretches(result.lambda_0, result.eta_0, r_arr)
    maxwell = alpha_p / (lambda_3 * lambda_3)
    sigma_1 = np.power(lambda_1, alpha) - np.power(lambda_3, alpha) - maxwell
    sigma_2 = np.power(lambda_2, alpha) - np.power(lambda_3, alpha) - maxwell
    return {
        "r": r_arr,
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
        "lambda_0": result.lambda_0,
        "eta_0": result.eta_0,
    }


def detect_wrinkling(stress_result: dict, tau: float = 0.01, min_width: float = 0.02) -> dict:
    """Detect wrinkling from a compute_stress result.

    Wrinkling on a stress field sigma_i is detected when there is a
    contiguous interval of r-coordinates over which sigma_i < -tau, AND
    that interval has width (r_high - r_low) >= min_width. Both sigma_1
    and sigma_2 are checked independently.

    Returns:
        {
          "detected": bool,
          "sigma_1_intervals": list of (r_low, r_high, r_mid) tuples,
          "sigma_2_intervals": list of (r_low, r_high, r_mid) tuples,
        }
        r_mid is the midpoint of each interval, rounded to 2 decimals.
    """
    r = np.asarray(stress_result["r"], dtype=float)

    def _intervals(sigma_field: np.ndarray) -> list:
        mask = np.asarray(sigma_field, dtype=float) < -tau
        out: list = []
        n = mask.size
        i = 0
        while i < n:
            if mask[i]:
                run_start = i
                while i < n and mask[i]:
                    i += 1
                run_end = i - 1
                r_low = float(r[run_start])
                r_high = float(r[run_end])
                if (r_high - r_low) >= min_width:
                    out.append((r_low, r_high, round(0.5 * (r_low + r_high), 2)))
            else:
                i += 1
        return out

    sigma_1_intervals = _intervals(stress_result["sigma_1"])
    sigma_2_intervals = _intervals(stress_result["sigma_2"])
    return {
        "detected": bool(sigma_1_intervals or sigma_2_intervals),
        "sigma_1_intervals": sigma_1_intervals,
        "sigma_2_intervals": sigma_2_intervals,
    }


def compute_max_electric_field(lambda_0: float) -> float:
    """Compute the dimensionless peak normalized electric field for the
    voltage-controlled scenario.

    Per section 4.4 of the source paper, the peak normalized electric field
    in the voltage-controlled scenario occurs at the pole of the membrane
    (r = 0) and is given by Ē_max = lambda_0^2, where Ē is the normalized
    electric field Ē = E h_0 / phi_0.

    Parameters
    ----------
    lambda_0 : float
        The Ritz amplitude for lambda_2 at the equilibrium.

    Returns
    -------
    float
        The dimensionless peak normalized electric field.
    """
    return float(lambda_0) ** 2


def plot_pcrit_envelope(ax: Any | None = None) -> Any:
    """Filled contour of ``p_crit`` over (alpha, alpha_p) with training points overlaid."""
    p_model, _ = load_envelope_models()
    alphas = np.linspace(1.2, 2.0, 60)
    alpha_ps = np.linspace(0.0, 0.72, 60)
    aa, ap = np.meshgrid(alphas, alpha_ps, indexing="xy")
    grid_pred = predict_p_crit(p_model, aa.ravel(), ap.ravel()).reshape(aa.shape)

    ax = _ensure_axes(ax)
    cs = ax.contourf(aa, ap, grid_pred, levels=20)
    ax.set_xlabel("alpha")
    ax.set_ylabel("alpha_p")
    ax.set_title("p_crit envelope")
    try:
        import pandas as pd
        from sembra.data import load_full_dataset, split_stable_unstable
        _, limit = split_stable_unstable(load_full_dataset())
        ax.scatter(limit["alpha"], limit["alpha_p"], c="white", s=10, edgecolors="k", label="training")
        ax.legend(loc="upper left")
    except Exception:
        pass
    fig = ax.figure
    fig.colorbar(cs, ax=ax, label="p_crit")
    return ax
