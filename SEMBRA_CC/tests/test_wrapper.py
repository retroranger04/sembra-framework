"""Tests for sembra_cc.wrapper.

These tests assume the full pipeline (network + envelope GPs) is trained
and available. Tests that exercise the live models require the production
artifacts and are skipped if those are missing.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sembra_cc import wrapper
from sembra_cc.envelope import LAMBDA_LIMIT_PATH, P_CRIT_PATH
from sembra_cc.network import NETWORK_PATH
from sembra_cc.wrapper import PredictionResult, predict


_artifacts_missing = not (
    Path(NETWORK_PATH).exists()
    and Path(P_CRIT_PATH).exists()
    and Path(LAMBDA_LIMIT_PATH).exists()
)

skip_if_no_artifacts = pytest.mark.skipif(
    _artifacts_missing,
    reason="trained artifacts (network.pt, envelope pickles) not yet produced",
)


@skip_if_no_artifacts
def test_typical_stable_query():
    r = predict(1.6, 0.10, 0.5)
    assert isinstance(r, PredictionResult)
    assert r.stable is True
    assert math.isfinite(r.lambda_0) and math.isfinite(r.eta_0)
    assert math.isfinite(r.det_H)


@skip_if_no_artifacts
def test_trivial_state_p_below_threshold():
    r = predict(1.6, 0.10, 0.0)
    assert r.lambda_0 == 1.0 and r.eta_0 == 0.0
    assert r.stable is True
    assert "Trivial undeformed state" in r.message


@skip_if_no_artifacts
def test_past_limit_query():
    # large p far above any plausible p_crit
    r = predict(1.6, 0.10, 5.0)
    assert r.stable is False
    assert math.isfinite(r.lambda_0)


def test_value_error_on_nan():
    with pytest.raises(ValueError):
        predict(float("nan"), 0.1, 0.5)


def test_value_error_on_inf():
    with pytest.raises(ValueError):
        predict(1.5, float("inf"), 0.5)


def test_value_error_on_alpha_too_low():
    with pytest.raises(ValueError):
        predict(1.0, 0.1, 0.5)


def test_value_error_on_alpha_too_high():
    with pytest.raises(ValueError):
        predict(2.5, 0.1, 0.5)


def test_value_error_on_negative_alpha_p():
    with pytest.raises(ValueError):
        predict(1.5, -0.01, 0.5)


def test_value_error_on_negative_p():
    with pytest.raises(ValueError):
        predict(1.5, 0.1, -0.01)


def test_value_error_on_bool_input():
    with pytest.raises(ValueError):
        predict(True, 0.1, 0.5)


@skip_if_no_artifacts
def test_grey_zone_returns_unstable():
    # Find a (alpha, alpha_p) and set p exactly equal to its p_crit estimate.
    from sembra_cc.envelope import load_envelope_models, predict_p_crit

    gp_p, _ = load_envelope_models()
    p_crit = float(predict_p_crit(gp_p, 1.6, 0.2))
    if p_crit <= 0:
        pytest.skip("p_crit non-positive at chosen point")
    r = predict(1.6, 0.2, p_crit)
    assert r.stable is False
    assert "0.5%" in r.message or "limit point" in r.message
