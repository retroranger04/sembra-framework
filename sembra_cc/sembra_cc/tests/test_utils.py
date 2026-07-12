"""Tests for the charge-controlled utility helpers."""

import numpy as np

from sembra_cc.calibrations import STRESS_DISPLAY_GRID_SIZE
from sembra_cc.utils import (
    compute_max_electric_field,
    compute_stress,
    detect_wrinkling,
)


def test_compute_stress_pole_value():
    """At r=0 (equibiaxial pole) t1 == t2 == the closed-form CC value.

    lam1 = lam2 = lambda_0 and lam3 = 1/lambda_0^2 there, so (Eq. 18)
        t_max = lambda_0**alpha - 1/lambda_0**(2*alpha) - Phi/lambda_0**4.
    """
    alpha, Phi = 1.6, 0.2
    lambda_0 = 1.5
    state = (lambda_0, 0.03, 0.3, -0.02)
    r_grid = np.linspace(0.0, 1.0, STRESS_DISPLAY_GRID_SIZE)

    stress = compute_stress(alpha, Phi, 0.4, state, r_grid)
    t_max = lambda_0**alpha - 1.0 / lambda_0 ** (2 * alpha) - Phi / lambda_0**4

    assert stress["t_1"].shape == r_grid.shape
    np.testing.assert_allclose(stress["t_1"][0], t_max, rtol=1e-10)
    np.testing.assert_allclose(stress["t_2"][0], t_max, rtol=1e-10)


def test_compute_max_electric_field_cc():
    """CC E_max = 1/sqrt((3 - a - 2*lambda_0)**2 + (b + 2*eta_0)**2)."""
    state = (1.4, 0.05, 0.3, -0.05)
    lambda_0, a, eta_0, b = state
    expected = 1.0 / np.sqrt((3 - a - 2 * lambda_0) ** 2 + (b + 2 * eta_0) ** 2)
    np.testing.assert_allclose(compute_max_electric_field(state), expected, rtol=1e-12)


def test_detect_wrinkling_positive_stress():
    """All-positive stresses flag no wrinkling."""
    r = np.linspace(0.0, 1.0, 401)
    stress = {"t_1": np.ones_like(r), "t_2": np.ones_like(r)}
    out = detect_wrinkling(stress, r)
    assert out["detected"] is False
    assert out["t_1_intervals"] == []
    assert out["t_2_intervals"] == []


def test_detect_wrinkling_negative_region():
    """A negative band in t_2 is detected and its interval reported."""
    r = np.linspace(0.0, 1.0, 401)
    t_2 = np.ones_like(r)
    t_2[(r >= 0.4) & (r <= 0.6)] = -1.0
    stress = {"t_1": np.ones_like(r), "t_2": t_2}
    out = detect_wrinkling(stress, r)
    assert out["detected"] is True
    assert out["t_1_intervals"] == []
    assert len(out["t_2_intervals"]) == 1
    r_start, r_end, r_mid = out["t_2_intervals"][0]
    assert r_start >= 0.39 and r_end <= 0.61
    assert 0.45 < r_mid < 0.55


# ---------------------------------------------------------------------------
# Notebook helpers: report formatting and the two figures
# ---------------------------------------------------------------------------
import math

import matplotlib

matplotlib.use("Agg")  # headless rendering for the figure tests

from sembra_cc.utils import (  # noqa: E402
    format_prediction_report,
    plot_design_space,
    plot_stress_field,
)
from sembra_cc.wrapper import PredictionResult  # noqa: E402


def _stable_result():
    """Synthetic stable-branch PredictionResult with realistic diagnostics."""
    alpha, Phi, p = 1.6, 0.1, 0.4
    state = (1.3, 0.02, 0.25, -0.03)
    r = np.linspace(0.0, 1.0, STRESS_DISPLAY_GRID_SIZE)
    stress = compute_stress(alpha, Phi, p, state, r)
    diagnostics = {
        "branch": "stable",
        "p_crit": 0.6,
        "lambda_limit": 1.8,
        "residual_norm": 2.3e-4,
        "det_H": 1.234567,
        "t_1": stress["t_1"],
        "t_2": stress["t_2"],
        "E_max": compute_max_electric_field(state),
        "wrinkling_detected": detect_wrinkling(stress, r)["detected"],
    }
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=state[0], a=state[1], eta_0=state[2], b=state[3],
        det_H=1.234567, stable=True, message="stable", diagnostics=diagnostics,
    )


def _no_equilibrium_result():
    """Synthetic no_equilibrium PredictionResult (state at the p_crit boundary)."""
    alpha, Phi, p = 1.6, 0.0, 5.0
    p_crit = 0.42
    state = (1.75, 0.04, 0.30, -0.04)
    r = np.linspace(0.0, 1.0, STRESS_DISPLAY_GRID_SIZE)
    stress = compute_stress(alpha, Phi, p_crit, state, r)
    diagnostics = {
        "branch": "no_equilibrium",
        "p_crit": p_crit,
        "lambda_limit": 1.75,
        "residual_norm": 5.1e-4,
        "det_H": -0.5,
        "t_1": stress["t_1"],
        "t_2": stress["t_2"],
        "E_max": compute_max_electric_field(state),
        "wrinkling_detected": detect_wrinkling(stress, r)["detected"],
    }
    return PredictionResult(
        alpha=alpha, Phi=Phi, p=p,
        lambda_0=state[0], a=state[1], eta_0=state[2], b=state[3],
        det_H=-0.5, stable=False, message="no_equilibrium", diagnostics=diagnostics,
    )


def _no_stable_region_result():
    """Synthetic no_stable_region PredictionResult (no valid state)."""
    diagnostics = {
        "branch": "no_stable_region",
        "p_crit": -0.1,
        "lambda_limit": math.nan,
        "residual_norm": math.nan,
        "det_H": math.nan,
        "t_1": None,
        "t_2": None,
        "E_max": math.nan,
        "wrinkling_detected": None,
    }
    return PredictionResult(
        alpha=1.6, Phi=0.3, p=0.2,
        lambda_0=math.nan, a=math.nan, eta_0=math.nan, b=math.nan,
        det_H=math.nan, stable=False, message="no_stable_region",
        diagnostics=diagnostics,
    )


def test_report_stable_branch_labels_and_residual_framing():
    """Stable report: scenario title, NOT DETECTED verdict, reframed residual."""
    report = format_prediction_report(_stable_result())
    assert "SEMBRA — Charge-Controlled" in report
    assert "Electromechanical instability:  NOT DETECTED ✓" in report
    # Correction 2: realistic reference range, not the projection tolerance.
    assert "(deployed surrogate typical range: 1e-4 to 1e-3)" in report
    assert "(ideal: < 1e-6)" not in report
    # All four Ritz parameters are present.
    for label in ("lambda_0", "a", "eta_0", "b", "det_H"):
        assert label in report


def test_report_no_equilibrium_note_and_verdict():
    """no_equilibrium report: DETECTED verdict + snap-through boundary note."""
    report = format_prediction_report(_no_equilibrium_result())
    assert "Electromechanical instability:  DETECTED ✗" in report
    assert "snap-through boundary equilibrium at p_crit" in report
    # Requested p is above p_crit, so the pressure difference is positive.
    assert "pressure difference = +" in report


def test_report_no_stable_region_is_nan_safe():
    """no_stable_region report renders N/A placeholders without raising."""
    report = format_prediction_report(_no_stable_region_result())
    assert "lambda_0 = N/A" in report
    assert "Ē_max = N/A" in report
    assert "‖∇Π‖ = N/A" in report


def test_plot_stress_field_returns_figure_and_saves(tmp_path, monkeypatch):
    """plot_stress_field returns a Figure and writes the deterministic PNG."""
    import sembra_cc.utils as utils_mod

    monkeypatch.setattr(utils_mod, "_query_plots_dir", lambda: tmp_path)
    result = _stable_result()
    fig = plot_stress_field(result)
    assert isinstance(fig, matplotlib.figure.Figure)
    expected = tmp_path / "stress_alpha_1.600_Phi_0.100_p_0.400.png"
    assert expected.exists()
    matplotlib.pyplot.close(fig)


def test_plot_design_space_returns_figure_and_saves(tmp_path, monkeypatch):
    """plot_design_space returns a Figure, writes the PNG, stars the query."""
    import sembra_cc.utils as utils_mod

    monkeypatch.setattr(utils_mod, "_query_plots_dir", lambda: tmp_path)
    result = _stable_result()
    fig = plot_design_space(result)
    assert isinstance(fig, matplotlib.figure.Figure)
    expected = tmp_path / "design_alpha_1.600_Phi_0.100_p_0.400.png"
    assert expected.exists()
    matplotlib.pyplot.close(fig)


def test_plot_design_space_no_equilibrium_headroom(tmp_path, monkeypatch):
    """no_equilibrium query (p > p_crit) leaves the star visible with headroom."""
    import sembra_cc.utils as utils_mod

    monkeypatch.setattr(utils_mod, "_query_plots_dir", lambda: tmp_path)
    result = _no_equilibrium_result()
    fig = plot_design_space(result)
    ax = fig.axes[0]
    # y-axis upper limit gives the star at least 10% headroom above it.
    assert ax.get_ylim()[1] >= result.p * 1.1 - 1e-9
    matplotlib.pyplot.close(fig)
