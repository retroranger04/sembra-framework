"""Tests for sembra_cc.utils."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sembra_cc.envelope import LAMBDA_LIMIT_PATH, P_CRIT_PATH
from sembra_cc.network import NETWORK_PATH
from sembra_cc.utils import (
    compute_stress,
    _ritz_stretches,
    detect_wrinkling,
    compute_max_electric_field,
)


_artifacts_missing = not (
    Path(NETWORK_PATH).exists()
    and Path(P_CRIT_PATH).exists()
    and Path(LAMBDA_LIMIT_PATH).exists()
)
skip_if_no_artifacts = pytest.mark.skipif(
    _artifacts_missing, reason="trained artifacts not yet produced"
)


@skip_if_no_artifacts
def test_compute_stress_keys_and_shapes():
    r = np.linspace(0.0, 1.0, 41)
    out = compute_stress(1.6, 0.10, 0.5, r)
    assert set(out.keys()) == {"r", "sigma_1", "sigma_2", "lambda_0", "eta_0"}
    assert out["r"].shape == (41,)
    assert out["sigma_1"].shape == (41,)
    assert out["sigma_2"].shape == (41,)


@skip_if_no_artifacts
def test_compute_stress_matches_hand_formula():
    """Recompute sigma directly from Section 4.7 (not via compute_stress) and
    confirm compute_stress's output matches at every r."""
    alpha, alpha_p, p = 1.6, 0.10, 0.5
    r = np.linspace(0.0, 1.0, 21)
    out = compute_stress(alpha, alpha_p, p, r)
    l1, l2, l3 = _ritz_stretches(out["lambda_0"], out["eta_0"], r)
    maxwell = alpha_p * (l3**2)
    s1 = l1**alpha - l3**alpha - maxwell
    s2 = l2**alpha - l3**alpha - maxwell
    assert np.allclose(out["sigma_1"], s1)
    assert np.allclose(out["sigma_2"], s2)


@skip_if_no_artifacts
def test_compute_stress_raises_on_nan_state():
    # alpha_p high enough that p_crit may be very small, then ask for a high p.
    # We rely on the wrapper returning NaN when p_crit <= 0; if not triggered,
    # build a contrived path: monkey-patch predict to return NaN state.
    import sembra_cc.utils as utils_mod
    from sembra_cc.wrapper import PredictionResult

    def fake_predict(alpha, alpha_p, p):
        return PredictionResult(
            alpha=alpha,
            alpha_p=alpha_p,
            p=p,
            lambda_0=float("nan"),
            eta_0=float("nan"),
            det_H=float("nan"),
            stable=False,
            message="fake nan",
        )

    real_predict = utils_mod.predict
    utils_mod.predict = fake_predict
    try:
        with pytest.raises(ValueError):
            compute_stress(1.6, 0.10, 0.5, np.linspace(0, 1, 5))
    finally:
        utils_mod.predict = real_predict


def test_detect_wrinkling_no_compression():
    # All-positive stresses → no wrinkling.
    r = np.linspace(0.0, 1.0, 401)
    stress = {"r": r, "sigma_1": np.ones_like(r) * 0.5, "sigma_2": np.ones_like(r) * 0.5}
    result = detect_wrinkling(stress)
    assert result["detected"] is False
    assert result["sigma_1_intervals"] == []
    assert result["sigma_2_intervals"] == []


def test_detect_wrinkling_sigma_2_at_rim():
    # sigma_2 below -tau over r in [0.95, 1.0] → wrinkling detected on sigma_2 only.
    r = np.linspace(0.0, 1.0, 401)
    sigma_1 = np.ones_like(r) * 0.5
    sigma_2 = np.where(r >= 0.95, -0.1, 0.5)
    stress = {"r": r, "sigma_1": sigma_1, "sigma_2": sigma_2}
    result = detect_wrinkling(stress)
    assert result["detected"] is True
    assert result["sigma_1_intervals"] == []
    assert len(result["sigma_2_intervals"]) == 1
    r_low, r_high, r_mid = result["sigma_2_intervals"][0]
    assert r_low >= 0.945 and r_low <= 0.955
    assert r_high >= 0.995 and r_high <= 1.001
    assert r_mid == 0.97 or r_mid == 0.98


def test_detect_wrinkling_below_min_width():
    # sigma_1 negative below -tau but only at a single point → no wrinkling.
    r = np.linspace(0.0, 1.0, 401)
    sigma_1 = np.ones_like(r) * 0.5
    sigma_1[200] = -0.2
    sigma_2 = np.ones_like(r) * 0.5
    result = detect_wrinkling({"r": r, "sigma_1": sigma_1, "sigma_2": sigma_2})
    assert result["detected"] is False


def test_detect_wrinkling_below_tau():
    # sigma_1 negative but with magnitude smaller than tau → no wrinkling.
    r = np.linspace(0.0, 1.0, 401)
    sigma_1 = np.full_like(r, -0.005)  # |value| < tau=0.01
    sigma_2 = np.ones_like(r) * 0.5
    result = detect_wrinkling({"r": r, "sigma_1": sigma_1, "sigma_2": sigma_2})
    assert result["detected"] is False


def test_compute_max_electric_field_cc_undeformed():
    # At lambda_0 = 1.0, eta_0 = 0.0 (trivial state): Ē_max = 1 / sqrt(1) = 1.0
    result = compute_max_electric_field(1.0, 0.0)
    assert abs(result - 1.0) < 1e-12


def test_compute_max_electric_field_cc_deformed():
    # At lambda_0 = 1.5, eta_0 = 0.8: denom = sqrt(0 + 2.56) = 1.6, so Ē_max = 0.625
    result = compute_max_electric_field(1.5, 0.8)
    assert abs(result - 0.625) < 1e-12


def test_compute_max_electric_field_cc_general():
    # General case: lambda_0 = 1.4, eta_0 = 0.5
    # denom = sqrt((3-2.8)^2 + 4*0.25) = sqrt(0.04 + 1.0) = sqrt(1.04) ≈ 1.0198
    # Ē_max ≈ 1/1.0198 ≈ 0.98058
    import math
    expected = 1.0 / math.sqrt((3.0 - 2.8)**2 + 4.0 * 0.25)
    result = compute_max_electric_field(1.4, 0.5)
    assert abs(result - expected) < 1e-12
