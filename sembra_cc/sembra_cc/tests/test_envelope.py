"""Tests for the charge-controlled GP envelope module.

Confirms the fitted envelopes load, produce finite and physically plausible
predictions in the training range, support array inputs, and are cached.
"""

import numpy as np

from sembra_cc.envelope import (
    load_all_envelopes,
    load_envelopes,
    load_wrinkling_envelope,
    predict_lambda_limit,
    predict_p_crit,
    predict_p_wrinkle,
)

# Representative (alpha, Phi) points well inside the training range.
REP_POINTS = [(1.3, 0.05), (1.6, 0.20), (1.9, 0.35)]

# Wrinkling-envelope training support spans the full CC Phi range [0.1, 0.4];
# keep these representative points inside it so predictions stay reliable.
WRINKLE_POINTS = [(1.4, 0.15), (1.6, 0.25), (1.8, 0.35)]


def test_load_envelopes_import():
    """Both envelopes load without error."""
    p_crit_env, lambda_limit_env = load_envelopes()
    assert p_crit_env is not None
    assert lambda_limit_env is not None


def test_predictions_finite_and_plausible():
    """p_crit > 0 and 1 < lambda_limit < 5 at representative in-range points."""
    p_crit_env, lambda_limit_env = load_envelopes()
    for alpha, phi in REP_POINTS:
        p_crit = predict_p_crit(p_crit_env, alpha, phi)
        lambda_limit = predict_lambda_limit(lambda_limit_env, alpha, phi)
        assert np.isfinite(p_crit) and p_crit > 0.0
        assert np.isfinite(lambda_limit) and 1.0 < lambda_limit < 5.0


def test_array_prediction_shape():
    """Array inputs return an array of matching shape with finite entries."""
    p_crit_env, _ = load_envelopes()
    alphas = np.array([a for a, _ in REP_POINTS])
    phis = np.array([phi for _, phi in REP_POINTS])
    out = predict_p_crit(p_crit_env, alphas, phis)
    assert out.shape == (len(REP_POINTS),)
    assert np.all(np.isfinite(out))


def test_cached_load_identity():
    """The cached load returns the same objects on a second call."""
    first = load_envelopes()
    second = load_envelopes()
    assert first[0] is second[0]
    assert first[1] is second[1]


def test_wrinkling_envelope_loads():
    """The wrinkling envelope loads and is the third of load_all_envelopes()."""
    env = load_wrinkling_envelope()
    assert env is not None
    all_three = load_all_envelopes()
    assert len(all_three) == 3
    assert all_three[2] is env
    # load_envelopes keeps its two-tuple contract.
    assert all_three[:2] == load_envelopes()


def test_predict_p_wrinkle_finite_and_plausible():
    """p_wrinkle is finite, positive, and bounded within the training p range."""
    for alpha, phi in WRINKLE_POINTS:
        p_wrinkle = predict_p_wrinkle(alpha, phi)
        assert np.isfinite(p_wrinkle)
        assert 0.0 < p_wrinkle < 3.0


def test_predict_p_wrinkle_array_shape():
    """Array Phi at fixed alpha returns a matching-shape, finite, positive array."""
    phis = np.array([phi for _, phi in WRINKLE_POINTS])
    out = predict_p_wrinkle(1.6, phis)
    assert out.shape == (len(WRINKLE_POINTS),)
    assert np.all(np.isfinite(out)) and np.all(out > 0.0)


def test_wrinkling_cached_identity():
    """The cached wrinkling load returns the same object on a second call."""
    assert load_wrinkling_envelope() is load_wrinkling_envelope()
