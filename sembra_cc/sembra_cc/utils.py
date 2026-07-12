"""Failure-mode helper functions for the charge-controlled scenario.

Provides the principal-stress field, wrinkling detection, and the maximum
normalized electric field. These feed the wrapper diagnostics and the notebook
figures. The scenario (charge-controlled) is fixed at the package level, so the
electrostatic terms below are the CC forms.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from sembra_cc.calibrations import PHI_MAX, PHI_MIN, STRESS_DISPLAY_GRID_SIZE
from sembra_cc.envelope import load_envelopes, predict_p_crit, predict_p_wrinkle

# Scenario label shown in the notebook report header. The only scenario-specific
# string among this module's notebook helpers (see SEMBRA v2 supplemental
# correction 5: keep the ``predict``-style single-argument helper signatures and
# read the scenario from this module-level constant).
SCENARIO_TITLE = "SEMBRA — Charge-Controlled"

# Fixed width of the report block's divider bars.
REPORT_WIDTH = 62

# Stress-map colormaps. Two goals:
#   * No pure or near-white anywhere on the plot (correction G): white disappears
#     against the notebook's white background, hiding low-stress regions and the
#     legend. The sequential maps start at a saturated mid tint (not near-white),
#     and the diverging maps pass through a DARK-NEUTRAL midtone (#4d4d4d), never
#     white.
#   * sigma_1 and sigma_2 use distinct colour families (correction H) so the two
#     panels are identifiable at a glance: sigma_1 (meridional) = RED family,
#     sigma_2 (circumferential) = BLUE family, in both the positive (sequential)
#     and zero-crossing (diverging) cases.
# Built once at module load, not per call.
_SIGMA1_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sigma1_seq",
    ["#fc9272", "#fb6a4a", "#cb181d", "#67000d"],  # salmon -> deep red
    N=256,
)
_SIGMA2_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sigma2_seq",
    ["#9ecae1", "#4292c6", "#2171b5", "#08306b"],  # light blue -> deep blue
    N=256,
)
_SIGMA1_DIVERGING = LinearSegmentedColormap.from_list(
    "sigma1_div",
    ["#08306b", "#4d4d4d", "#a50f15"],  # blue -> dark gray -> dark red
    N=256,
)
_SIGMA2_DIVERGING = LinearSegmentedColormap.from_list(
    "sigma2_div",
    ["#7f2704", "#4d4d4d", "#08306b"],  # dark orange-brown -> dark gray -> dark blue
    N=256,
)

# Design-space figure region palette (paper Figures 6/7 three-region convention):
# muted and visually distinct, none white, the "safe" region reads most benign.
# The query-marker colour is chosen to contrast against all three region fills
# and the black boundary curves.
_DESIGN_SAFE_COLOR = "#a6d8a8"     # soft green  -- stable, no wrinkling
_DESIGN_WRINKLE_COLOR = "#f6d27a"  # soft amber  -- stable but wrinkling
_DESIGN_NOEQ_COLOR = "#e79a9a"     # soft red    -- no equilibrium
_DESIGN_QUERY_COLOR = "#0033cc"    # strong blue -- query marker / crosshair


def _kinematics(alpha, Phi, state, r):
    """Return the principal stretches (lam1, lam2, lam3) on the radial grid.

    ``state`` is ``(lambda_0, a, eta_0, b)``; ``r`` is the radial grid array. The
    expressions mirror the Ritz ansatz used by the symbolic core.
    """
    lambda_0, a, eta_0, b = state
    lam2 = r**2 + lambda_0 * (1 - r**2) + a * r**3 * (1 - r)
    lam2p = 2 * r * (1 - lambda_0) + a * (3 * r**2 - 4 * r**3)
    etap = -2 * eta_0 * r + 3 * b * r**2 - 4 * b * r**3
    lam1 = np.sqrt((lam2 + r * lam2p) ** 2 + etap**2)
    lam3 = 1.0 / (lam1 * lam2)
    return lam1, lam2, lam3


def compute_stress(alpha, Phi, p, state, r_grid=None) -> dict:
    """Normalized principal Cauchy stresses ``t1`` (meridional) and ``t2``.

    Charge-controlled form (paper Eq. 18), nondimensionalized by the shear
    modulus (mu = 1, so the returned values are dimensionless ``t_i = sigma_i/mu``):

        t1 = lam1**alpha - Phi * lam3**2 - lam3**alpha
        t2 = lam2**alpha - Phi * lam3**2 - lam3**alpha

    Evaluated on ``r_grid`` (defaults to the ``STRESS_DISPLAY_GRID_SIZE`` uniform
    grid on ``[0, 1]``). ``p`` does not enter the normalized stress and is
    accepted only for a uniform call signature. Returns a dict with ``'t_1'`` and
    ``'t_2'`` arrays.
    """
    if r_grid is None:
        r_grid = np.linspace(0.0, 1.0, STRESS_DISPLAY_GRID_SIZE)
    r_grid = np.asarray(r_grid, dtype=float)

    lam1, lam2, lam3 = _kinematics(alpha, Phi, state, r_grid)
    maxwell = Phi * lam3**2  # CC electrostatic contribution
    lam3_a = lam3**alpha
    t_1 = lam1**alpha - maxwell - lam3_a
    t_2 = lam2**alpha - maxwell - lam3_a
    return {"t_1": t_1, "t_2": t_2}


def _negative_intervals(r_grid, values):
    """Return contiguous ``(r_start, r_end, r_mid)`` intervals where values < 0."""
    negative = values < 0.0
    intervals = []
    start = None
    for i, is_neg in enumerate(negative):
        if is_neg and start is None:
            start = i
        elif not is_neg and start is not None:
            intervals.append((r_grid[start], r_grid[i - 1],
                              0.5 * (r_grid[start] + r_grid[i - 1])))
            start = None
    if start is not None:
        intervals.append((r_grid[start], r_grid[-1],
                          0.5 * (r_grid[start] + r_grid[-1])))
    return intervals


def detect_wrinkling(stress: dict, r_grid=None) -> dict:
    """Flag wrinkling where either principal stress drops below zero.

    ``stress`` is the dict from :func:`compute_stress`. Returns ``'detected'``
    (bool) and the negative ``r``-intervals for each principal stress as lists of
    ``(r_start, r_end, r_mid)`` tuples.
    """
    t_1 = np.asarray(stress["t_1"], dtype=float)
    t_2 = np.asarray(stress["t_2"], dtype=float)
    if r_grid is None:
        r_grid = np.linspace(0.0, 1.0, len(t_1))
    r_grid = np.asarray(r_grid, dtype=float)

    t1_intervals = _negative_intervals(r_grid, t_1)
    t2_intervals = _negative_intervals(r_grid, t_2)
    return {
        "detected": bool(t1_intervals or t2_intervals),
        "t_1_intervals": t1_intervals,
        "t_2_intervals": t2_intervals,
    }


def compute_max_electric_field(state) -> float:
    """Maximum normalized electric field ``E_max``.

    Charge-controlled form:
    ``E_max = 1 / sqrt((3 - a - 2*lambda_0)**2 + (b + 2*eta_0)**2)``.
    ``state`` is ``(lambda_0, a, eta_0, b)``.
    """
    lambda_0, a, eta_0, b = state
    return float(1.0 / np.sqrt((3 - a - 2 * lambda_0) ** 2 + (b + 2 * eta_0) ** 2))


# ---------------------------------------------------------------------------
# Notebook helpers: formatted report and the two figures
# ---------------------------------------------------------------------------
def _fmt_input(x) -> str:
    """Compact display of an input scalar (e.g. 1.6, 0.1, 5)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x):.6g}"


def _fmt6(x) -> str:
    """Fixed 6-decimal display; ``N/A`` for missing / NaN values."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x):.6f}"


def _fmt_sci(x) -> str:
    """Two-significant-figure scientific display; ``N/A`` for missing / NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x):.2e}"


def _emi_label(branch) -> str:
    """Electromechanical-instability verdict keyed on the router branch.

    Per supplemental correction 1 the branches are ``stable`` / ``near_fold`` /
    ``no_equilibrium`` (plus ``no_stable_region``).
    """
    if branch == "stable":
        return "NOT DETECTED ✓"
    if branch == "near_fold":
        return "MARGINAL — near fold ⚠"
    # no_equilibrium and no_stable_region both indicate instability.
    return "DETECTED ✗"


def _fmt_pressure_diff(p, p_crit) -> str:
    """Signed percentage distance of ``p`` from ``p_crit``."""
    if p_crit is None or not np.isfinite(p_crit) or p_crit == 0.0:
        return "N/A"
    return f"{(p - p_crit) / p_crit * 100.0:+.2f}%"


def _peak(values):
    """Return ``(peak_str, r_str)`` for the maximum of a stress array."""
    if values is None:
        return "N/A", "N/A"
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return "N/A", "N/A"
    r_grid = np.linspace(0.0, 1.0, arr.size)
    idx = int(np.nanargmax(arr))
    return f"{arr[idx]:.6f}", f"{r_grid[idx]:.4f}"


def _wrinkling_detail(result) -> str:
    """One-line summary of the wrinkling-affected intervals in the deformed ρ.

    The intervals are reported in the deformed radial coordinate ``ρ = r·λ₂(r)``
    (correction E) so they are consistent with the stress-map axis, which is also
    ρ. ``detect_wrinkling`` operates on the undeformed ``r`` grid; the endpoints are
    mapped to ρ using the equilibrium ``λ₂`` field.
    """
    d = result.diagnostics
    t_1 = d.get("t_1")
    t_2 = d.get("t_2")
    if t_1 is None or t_2 is None:
        return ""
    state = (result.lambda_0, result.a, result.eta_0, result.b)
    if not np.all(np.isfinite(state)):
        return ""
    r_grid = np.linspace(0.0, 1.0, len(t_1))
    wr = detect_wrinkling({"t_1": t_1, "t_2": t_2}, r_grid)
    if not wr["detected"]:
        return ""

    _lam1, lam2_grid, _lam3 = _kinematics(result.alpha, result.Phi, state, r_grid)

    def _rho(r):
        return float(r * np.interp(r, r_grid, lam2_grid))

    parts = []
    for r_s, r_e, _ in wr["t_1_intervals"]:
        parts.append(f"σ_1 ≤ 0 on ρ ∈ [{_rho(r_s):.3f}, {_rho(r_e):.3f}]")
    for r_s, r_e, _ in wr["t_2_intervals"]:
        parts.append(f"σ_2 ≤ 0 on ρ ∈ [{_rho(r_s):.3f}, {_rho(r_e):.3f}]")
    return "; ".join(parts)


def format_prediction_report(result) -> str:
    """Render the user-facing text report for a :class:`PredictionResult`.

    The block matches the SEMBRA v2 PRD layout with the supplemental corrections:
    branch-aware electromechanical-instability verdict (correction 1), a residual
    reference range instead of the projection tolerance (correction 2), and an
    explicit note when the reported state is the snap-through boundary rather than
    the requested ``p`` (correction 3). Returns a single string; the notebook
    prints it once (correction 6).
    """
    d = result.diagnostics
    branch = d.get("branch")
    bar = "=" * REPORT_WIDTH
    lines = [bar, SCENARIO_TITLE.center(REPORT_WIDTH), bar, ""]

    lines += [
        "  Input",
        f"    alpha    = {_fmt_input(result.alpha)}",
        f"    Phi      = {_fmt_input(result.Phi)}",
        f"    p        = {_fmt_input(result.p)}",
        "",
    ]

    lines += [
        "  Equilibrium state",
        f"    lambda_0 = {_fmt6(result.lambda_0)}",
        f"    a        = {_fmt6(result.a)}",
        f"    eta_0    = {_fmt6(result.eta_0)}",
        f"    b        = {_fmt6(result.b)}",
        f"    det_H    = {_fmt6(result.det_H)}",
        "",
    ]
    if branch == "no_equilibrium":
        margin = d.get("boundary_margin")
        where = (
            "at p_crit" if margin is None
            else f"at {margin * 100:.1f}% below p_crit for numerical convergence"
        )
        lines += [
            f"    (state shown is the snap-through boundary equilibrium {where};",
            "     not the requested p; no static equilibrium exists at requested p)",
            "",
        ]

    lines.append("  Failure modes")
    lines.append(f"    Electromechanical instability:  {_emi_label(branch)}")
    p_crit = d.get("p_crit")
    lines.append(
        f"      p_crit ≈ {_fmt_input(p_crit)},  "
        f"pressure difference = {_fmt_pressure_diff(result.p, p_crit)}"
    )

    wr = d.get("wrinkling_detected")
    if wr is None:
        wr_label = "N/A"
    else:
        wr_label = "DETECTED ✗" if wr else "NOT DETECTED ✓"
    lines.append(f"    Wrinkling instability:      {wr_label}")
    detail = _wrinkling_detail(result)
    if detail:
        lines.append(f"      {detail}")

    lines.append(f"    Peak electric field:        Ē_max = {_fmt6(d.get('E_max'))}")
    s1 = _peak(d.get("t_1"))
    s2 = _peak(d.get("t_2"))
    lines.append("    Peak stresses")
    lines.append(f"      σ_1_max = {s1[0]} at r ≈ {s1[1]}")
    lines.append(f"      σ_2_max = {s2[0]} at r ≈ {s2[1]}")
    lines.append("")

    lines.append(
        f"  Residual: ‖∇Π‖ = {_fmt_sci(d.get('residual_norm'))} "
        "(deployed surrogate typical range: 1e-4 to 1e-3)"
    )
    lines.append("")
    lines.append(bar)
    return "\n".join(lines)


def _query_plots_dir() -> Path:
    """Directory where per-query figures are written (created on first use)."""
    out_dir = Path(__file__).resolve().parent.parent / "logs" / "plots" / "user_queries"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_figure(fig, kind, result) -> Path:
    """Save ``fig`` under a deterministic per-query filename and return the path."""
    fname = (
        f"{kind}_alpha_{result.alpha:.3f}"
        f"_Phi_{result.Phi:.3f}_p_{result.p:.3f}.png"
    )
    path = _query_plots_dir() / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _stress_title(result) -> str:
    """Branch-aware suptitle for the stress figure (supplemental correction 3)."""
    if result.diagnostics.get("branch") == "no_equilibrium":
        p_crit = result.diagnostics.get("p_crit")
        return (
            f"Stress field at snap-through threshold "
            f"(α={result.alpha:g}, Φ={result.Phi:g}, p_crit={p_crit:.4f})"
        )
    return (
        f"Principal stresses on the deformed membrane "
        f"(α={result.alpha:g}, Φ={result.Phi:g}, p={result.p:g})"
    )


def plot_stress_field(result):
    """Two-panel principal-stress field on the mirrored deformed membrane.

    Left panel σ₁ (meridional, red family), right panel σ₂ (circumferential, blue
    family), each drawn as a stress-coloured ``LineCollection`` along the membrane
    cross-section ``ρ(r) = r·λ₂(r)``, ``η(r) = η₀(1−r²) + b·r³(1−r)``, mirrored
    across the symmetry axis. Colormaps avoid white entirely (sequential for
    strictly-positive fields, dark-neutral-centred diverging for zero-crossing
    fields). Peak values annotated; compressive band shaded when present. Returns
    the :class:`matplotlib.figure.Figure` and saves it to ``logs/plots/user_queries/``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    state = (result.lambda_0, result.a, result.eta_0, result.b)
    t_1 = result.diagnostics.get("t_1")
    t_2 = result.diagnostics.get("t_2")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    if t_1 is None or t_2 is None or not np.all(np.isfinite(state)):
        for ax in axes:
            ax.axis("off")
        axes[0].text(
            0.5, 0.5,
            "No equilibrium state available;\nstress field cannot be shown.",
            ha="center", va="center", transform=axes[0].transAxes,
        )
        fig.suptitle(_stress_title(result))
        _save_figure(fig, "stress", result)
        return fig

    r = np.linspace(0.0, 1.0, len(t_1))
    _lam1, lam2, _lam3 = _kinematics(result.alpha, result.Phi, state, r)
    rho = r * lam2
    eta = result.eta_0 * (1 - r**2) + result.b * r**3 * (1 - r)

    panels = [
        (axes[0], np.asarray(t_1, dtype=float), _SIGMA1_SEQUENTIAL, _SIGMA1_DIVERGING,
         "σ₁ (meridional)"),
        (axes[1], np.asarray(t_2, dtype=float), _SIGMA2_SEQUENTIAL, _SIGMA2_DIVERGING,
         "σ₂ (circumferential)"),
    ]
    for ax, vals, cmap_seq, cmap_div, label in panels:
        # Mirror the quarter cross-section across the symmetry axis so the user
        # sees the full membrane cross-section.
        full_rho = np.concatenate([-rho[::-1], rho])
        full_eta = np.concatenate([eta[::-1], eta])
        full_vals = np.concatenate([vals[::-1], vals])
        pts = np.column_stack([full_rho, full_eta])
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        seg_vals = 0.5 * (full_vals[:-1] + full_vals[1:])

        vmin_data = float(np.min(vals))
        vmax_data = float(np.max(vals))
        # Guard against degenerate ranges (constant field or NaN).
        if (not np.isfinite(vmin_data)
                or not np.isfinite(vmax_data)
                or vmax_data - vmin_data < 1e-12):
            vmin_data, vmax_data = 0.0, 1.0
        if vmin_data >= 0.0:
            # Strictly positive field: sequential colormap over the actual data
            # range — no wasted color budget, no misleading white.
            cmap_used = cmap_seq
            norm = Normalize(vmin_data, vmax_data)
        else:
            # Field crosses zero (e.g. wrinkling): white-free custom diverging
            # colormap centered at zero so the sign change reads distinctly and
            # nothing disappears against the notebook's white background.
            cmap_used = cmap_div
            extent = max(abs(vmin_data), abs(vmax_data))
            norm = Normalize(-extent, extent)
        lc = LineCollection(segs, cmap=cmap_used, norm=norm)
        lc.set_array(seg_vals)
        lc.set_linewidth(3.0)
        ax.add_collection(lc)
        ax.set_xlim(full_rho.min() * 1.05, full_rho.max() * 1.05)
        ax.set_ylim(min(0.0, full_eta.min()) * 1.05, full_eta.max() * 1.05 + 1e-9)
        ax.set_aspect("equal")
        ax.set_xlabel("ρ")
        ax.set_ylabel("η")
        ax.set_title(label)
        cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("stress / μ")

        idx = int(np.argmax(vals))
        ax.annotate(
            f"peak = {vals[idx]:.3f}\nat r ≈ {r[idx]:.3f}",
            xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
        )

        # Subtle location band where the field is compressive. The wrinkling
        # verdict itself is communicated by the report block and by the negative
        # side of the diverging colormap, so no redundant text overlay here
        # (correction D).
        neg_mask = vals < 0.0
        if neg_mask.any():
            rlo = float(rho[neg_mask].min())
            rhi = float(rho[neg_mask].max())
            ax.axvspan(rlo, rhi, color="0.5", alpha=0.15)
            ax.axvspan(-rhi, -rlo, color="0.5", alpha=0.15)

    fig.suptitle(_stress_title(result))
    fig.tight_layout()
    _save_figure(fig, "stress", result)
    return fig


def _mark_query_point(ax, phi, p, y_lower):
    """Draw the precise query marker: small circle, crosshair guides, coordinates."""
    ax.plot([phi], [p], marker="o", markersize=7, color=_DESIGN_QUERY_COLOR,
            markeredgecolor="white", markeredgewidth=0.8, linestyle="none",
            zorder=6, label="your query (Φ, p)")
    # Thin dotted crosshair from the marker to each axis pins the exact coordinates.
    ax.plot([phi, phi], [y_lower, p], color=_DESIGN_QUERY_COLOR, lw=0.8,
            linestyle=":", alpha=0.8, zorder=5)
    ax.plot([PHI_MIN, phi], [p, p], color=_DESIGN_QUERY_COLOR, lw=0.8,
            linestyle=":", alpha=0.8, zorder=5)
    ax.annotate(
        f"(Φ={phi:g}, p={p:g})", xy=(phi, p), xytext=(6, 6),
        textcoords="offset points", fontsize=8, color=_DESIGN_QUERY_COLOR, zorder=7,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_DESIGN_QUERY_COLOR,
                  alpha=0.85, lw=0.6),
    )


def _design_space_no_stable_fallback(fig, ax, result, phi_grid, p_wrinkle_curve):
    """Graceful render when the fold curve is undefined across the whole α slice.

    Draws the wrinkling boundary alone if it is defined anywhere, otherwise a
    short message figure. Saves and returns the figure like the main path.
    """
    star_phi, star_p = result.Phi, result.p
    wrink_mask = np.isfinite(p_wrinkle_curve) & (p_wrinkle_curve > 0.0)
    if wrink_mask.any():
        y_upper = float(np.nanmax(p_wrinkle_curve[wrink_mask])) * 1.3
        if np.isfinite(star_p):
            y_upper = max(y_upper, star_p * 1.1)
        ax.plot(phi_grid, np.where(wrink_mask, p_wrinkle_curve, np.nan),
                color="black", lw=1.8, linestyle="--", label=r"$\min(t_2) = 0$")
        _mark_query_point(ax, star_phi, star_p, 0.0)
        ax.set_xlim(PHI_MIN, PHI_MAX)
        ax.set_ylim(0.0, y_upper)
        ax.set_xlabel("Φ")
        ax.set_ylabel("p")
        ax.set_title(f"Design space at α = {result.alpha:g}  (no stable region)")
        ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5,
                f"No stable region and no wrinkling boundary\ndefined at α = {result.alpha:g}.",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    _save_figure(fig, "design", result)
    return fig


def plot_design_space(result):
    """Design-space location of the query in the ``(Φ, p)`` plane at the user's α.

    Renders the paper's Figures 6/7 three-region convention: the fold boundary
    ``det(H)=0`` (solid black) and the wrinkling boundary ``t_2=0`` (dashed black),
    with three shaded regions — stable-no-wrinkling (safe), stable-with-wrinkling,
    and no-equilibrium — plus a precise query marker (small circle + crosshair +
    numeric coordinates). Handles the ``no_equilibrium`` case (query above the fold
    curve) with y-axis headroom and quadrant-aware legend placement, and falls back
    gracefully when the fold curve is undefined across the whole slice. Returns the
    figure and saves it to ``logs/plots/user_queries/``.
    """
    import matplotlib.pyplot as plt

    p_crit_envelope = load_envelopes()[0]
    phi_grid = np.linspace(PHI_MIN, PHI_MAX, STRESS_DISPLAY_GRID_SIZE)
    p_crit_curve = np.asarray(
        predict_p_crit(p_crit_envelope, result.alpha, phi_grid), dtype=float
    )
    p_wrinkle_curve = np.asarray(predict_p_wrinkle(result.alpha, phi_grid), dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    star_phi, star_p = result.Phi, result.p

    valid_fold = np.isfinite(p_crit_curve) & (p_crit_curve > 0.0)
    if not valid_fold.any():
        # no_stable_region: the fold curve is undefined across the entire slice.
        return _design_space_no_stable_fallback(
            fig, ax, result, phi_grid, p_wrinkle_curve
        )

    # y-axis: headroom above the fold curve and the query point (correction 9).
    y_lower = 0.0
    y_upper = float(np.nanmax(p_crit_curve[valid_fold])) * 1.15
    if np.isfinite(star_p):
        y_upper = max(y_upper, star_p * 1.1)

    # The dashed t_2=0 curve is physical only where it is finite, positive, and
    # below the fold curve (paper ordering: t_2=0 sits under det(H)=0); we draw the
    # curve only there. For the SHADING, clamp the wrinkling band edge into
    # [0, p_crit] so the three regions stay contiguous across the full Phi range
    # (correction C): without the clamp, the CC wrinkling GP extrapolating to <= 0
    # below its low-Phi training support flips the fill mask and paints an abrupt
    # green strip. Clamping makes the green sub-region shrink smoothly to zero
    # instead of jumping.
    wrink_curve_mask = (
        valid_fold
        & np.isfinite(p_wrinkle_curve)
        & (p_wrinkle_curve > 0.0)
        & (p_wrinkle_curve < p_crit_curve)
    )
    band_lower = np.where(
        np.isfinite(p_wrinkle_curve),
        np.clip(p_wrinkle_curve, 0.0, p_crit_curve),
        p_crit_curve,
    )

    # --- three-region shading (contiguous) --------------------------------------
    # green [band_lower, p_crit] = left of / above the t_2=0 wrinkling curve,
    #     below the fold (stable, no wrinkling);
    # amber [0, band_lower] = right of / below the t_2=0 wrinkling curve
    #     (stable + wrinkling);
    # red [p_crit, y_upper] = above the fold (no equilibrium).
    ax.fill_between(phi_grid, band_lower, p_crit_curve, where=valid_fold,
                    color=_DESIGN_SAFE_COLOR, alpha=0.75,
                    label="stable region")
    ax.fill_between(phi_grid, y_lower, band_lower, where=valid_fold,
                    color=_DESIGN_WRINKLE_COLOR, alpha=0.75,
                    label="wrinkling instability region")
    ax.fill_between(phi_grid, p_crit_curve, y_upper, where=valid_fold,
                    color=_DESIGN_NOEQ_COLOR, alpha=0.75,
                    label="electromechanical instability region")

    # --- boundary curves (paper Section 4 labels) -------------------------------
    ax.plot(phi_grid, np.where(valid_fold, p_crit_curve, np.nan),
            color="black", lw=2.0, label=r"$\det(H) = 0$")
    ax.plot(phi_grid, np.where(wrink_curve_mask, p_wrinkle_curve, np.nan),
            color="black", lw=1.8, linestyle="--", label=r"$\min(t_2) = 0$")

    # --- precise query marker ---------------------------------------------------
    _mark_query_point(ax, star_phi, star_p, y_lower)

    ax.set_xlim(PHI_MIN, PHI_MAX)
    ax.set_ylim(y_lower, y_upper)
    ax.set_xlabel("Φ")
    ax.set_ylabel("p")
    ax.set_title(f"Design space at α = {result.alpha:g}")

    # Quadrant-aware legend placement (correction 9): opposite the query marker so
    # it never overlaps it.
    x_frac = (star_phi - PHI_MIN) / (PHI_MAX - PHI_MIN) if PHI_MAX > PHI_MIN else 0.5
    y_frac = (star_p - y_lower) / (y_upper - y_lower) if y_upper > y_lower else 0.5
    vert = "lower" if y_frac > 0.5 else "upper"
    horiz = "left" if x_frac > 0.5 else "right"
    ax.legend(loc=f"{vert} {horiz}", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save_figure(fig, "design", result)
    return fig
