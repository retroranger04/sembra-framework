"""Visualization and stress-computation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sembra_cc.wrapper import predict


def _ritz_stretches(lambda_0: float, eta_0: float, r: np.ndarray):
    r = np.asarray(r, dtype=np.float64)
    lambda_2 = r**2 + lambda_0 * (1.0 - r**2)
    dlambda2_dr = 2.0 * r * (1.0 - lambda_0)
    deta_dr = -2.0 * eta_0 * r
    inner = lambda_2 + r * dlambda2_dr
    lambda_1 = np.sqrt(inner**2 + deta_dr**2)
    lambda_3 = 1.0 / (lambda_1 * lambda_2)
    return lambda_1, lambda_2, lambda_3


def compute_stress(
    alpha: float, alpha_p: float, p: float, r: np.ndarray
) -> dict:
    """Compute principal Cauchy stresses on the deformed meridian.

        sigma_1(r) = lambda_1(r)^alpha - lambda_3(r)^alpha - alpha_p * lambda_3(r)^2
        sigma_2(r) = lambda_2(r)^alpha - lambda_3(r)^alpha - alpha_p * lambda_3(r)^2
    """
    result = predict(alpha, alpha_p, p)
    if not np.isfinite(result.lambda_0) or not np.isfinite(result.eta_0):
        raise ValueError(
            f"compute_stress: predict returned non-finite state at "
            f"(alpha={alpha}, alpha_p={alpha_p}, p={p}). "
            f"Wrapper message: {result.message}"
        )

    r_arr = np.asarray(r, dtype=np.float64)
    lambda_1, lambda_2, lambda_3 = _ritz_stretches(
        result.lambda_0, result.eta_0, r_arr
    )
    maxwell = alpha_p * lambda_3**2
    sigma_1 = lambda_1**alpha - lambda_3**alpha - maxwell
    sigma_2 = lambda_2**alpha - lambda_3**alpha - maxwell

    return {
        "r": r_arr,
        "sigma_1": sigma_1,
        "sigma_2": sigma_2,
        "lambda_0": result.lambda_0,
        "eta_0": result.eta_0,
    }


def detect_wrinkling(
    stress_result: dict, tau: float = 0.01, min_width: float = 0.02
) -> dict:
    """Detect wrinkling from a compute_stress result.

    Wrinkling on a stress field sigma_i is detected when there is a
    contiguous interval of r-coordinates over which sigma_i < -tau, AND
    that interval has width (r_high - r_low) >= min_width. Both sigma_1
    and sigma_2 are checked independently.

    Parameters
    ----------
    stress_result : dict
        Dict returned by compute_stress, containing 'r', 'sigma_1', 'sigma_2'.
    tau : float
        Magnitude threshold for "clearly negative" stress. Default 0.01.
    min_width : float
        Minimum contiguous interval width in r-coordinates. Default 0.02.

    Returns
    -------
    dict
        {
          "detected": bool,
          "sigma_1_intervals": list of (r_low, r_high, r_mid) tuples,
          "sigma_2_intervals": list of (r_low, r_high, r_mid) tuples,
        }
        r_mid is the midpoint of each interval, rounded to 2 decimals.
    """
    r = np.asarray(stress_result["r"], dtype=np.float64)
    if r.size < 2:
        return {"detected": False, "sigma_1_intervals": [], "sigma_2_intervals": []}

    def _intervals_for(field: np.ndarray) -> list:
        mask = field < -tau
        out = []
        i = 0
        n = mask.size
        while i < n:
            if mask[i]:
                j = i
                while j + 1 < n and mask[j + 1]:
                    j += 1
                r_low = float(r[i])
                r_high = float(r[j])
                if r_high - r_low >= min_width:
                    r_mid = round((r_low + r_high) / 2.0, 2)
                    out.append((r_low, r_high, r_mid))
                i = j + 1
            else:
                i += 1
        return out

    s1_intervals = _intervals_for(np.asarray(stress_result["sigma_1"], dtype=np.float64))
    s2_intervals = _intervals_for(np.asarray(stress_result["sigma_2"], dtype=np.float64))
    return {
        "detected": bool(s1_intervals) or bool(s2_intervals),
        "sigma_1_intervals": s1_intervals,
        "sigma_2_intervals": s2_intervals,
    }


def compute_max_electric_field(lambda_0: float, eta_0: float) -> float:
    """Compute the dimensionless peak normalized electric field for the
    charge-controlled scenario.

    Per section 4.4 of the source paper, the peak normalized electric field
    in the charge-controlled scenario occurs at the clamped boundary (r = 1)
    of the membrane and is given by

        Ē_max = 1 / sqrt((3 - 2*lambda_0)^2 + 4*eta_0^2)

    where Ē is the normalized electric field Ē = E h_0 / phi_0.

    Parameters
    ----------
    lambda_0 : float
        The Ritz amplitude for lambda_2 at the equilibrium.
    eta_0 : float
        The Ritz amplitude for eta at the equilibrium.

    Returns
    -------
    float
        The dimensionless peak normalized electric field.
    """
    import math
    denom_sq = (3.0 - 2.0 * float(lambda_0)) ** 2 + 4.0 * float(eta_0) ** 2
    return 1.0 / math.sqrt(denom_sq)


def plot_shape(alpha: float, alpha_p: float, p: float, ax=None):
    """Plot the membrane shape eta(r) at the predicted equilibrium."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    result = predict(alpha, alpha_p, p)
    if not np.isfinite(result.lambda_0) or not np.isfinite(result.eta_0):
        ax.text(0.5, 0.5, "no stable state", ha="center", va="center")
        return ax
    r = np.linspace(0.0, 1.0, 401)
    eta = result.eta_0 * (1.0 - r**2)
    ax.plot(r, eta, color="C0", linewidth=2.0)
    ax.set_xlabel("r")
    ax.set_ylabel("η(r)")
    ax.set_title(
        f"Membrane shape  α={alpha}, α_p={alpha_p}, p={p}"
    )
    ax.grid(alpha=0.3)
    return ax


def plot_pcrit_envelope(ax=None):
    """Filled contour of p_crit over (alpha, alpha_p)."""
    import matplotlib.pyplot as plt

    from .data import load_full_dataset, split_stable_unstable
    from .envelope import load_envelope_models, predict_p_crit

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    gp_p, _ = load_envelope_models()
    alphas = np.linspace(1.2, 2.0, 80)
    alpha_ps = np.linspace(0.0, 0.40, 80)
    A, AP = np.meshgrid(alphas, alpha_ps)
    Z = predict_p_crit(gp_p, A, AP)
    Z = np.clip(Z, 0.0, None)
    cf = ax.contourf(A, AP, Z, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="p_crit")

    df = load_full_dataset()
    _, limit_df = split_stable_unstable(df)
    ax.scatter(
        limit_df["alpha"], limit_df["alpha_p"], s=12, color="white",
        edgecolor="black", linewidth=0.5, label="limit-point rows",
    )
    ax.set_xlabel("α")
    ax.set_ylabel("α_p")
    ax.set_title("p_crit envelope")
    ax.legend(loc="upper right")
    return ax
