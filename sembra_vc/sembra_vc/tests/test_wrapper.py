"""Tests for the voltage-controlled wrapper."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from sembra_vc import wrapper as wrapper_module
from sembra_vc.wrapper import PredictionResult, predict

_REQUIRED_DIAG_KEYS = {
    "p_crit", "lambda_limit", "branch", "residual_norm",
    "projection_converged", "projection_iterations",
}
_EXPECTED_FIELD_ORDER = [
    "alpha", "Phi", "p", "lambda_0", "a", "eta_0", "b",
    "det_H", "stable", "message", "diagnostics",
]


def test_field_order_matches_prd():
    assert [f.name for f in fields(PredictionResult)] == _EXPECTED_FIELD_ORDER


@pytest.mark.parametrize(
    "alpha, Phi, p",
    [
        (0.5, 0.1, 1.0),   # alpha below ALPHA_MIN
        (2.5, 0.1, 1.0),   # alpha above ALPHA_MAX
        (1.6, 0.9, 1.0),   # Phi above PHI_MAX
        (1.6, -0.1, 1.0),  # Phi below PHI_MIN
        (1.6, 0.1, -1.0),  # negative p
    ],
)
def test_validation_rejects_out_of_range(alpha, Phi, p):
    with pytest.raises(ValueError):
        predict(alpha, Phi, p)


@pytest.mark.parametrize("p", [1e-9, 0.0])
def test_trivial_p_shortcut(p):
    """p at or below the threshold returns the undeformed, stable state."""
    result = predict(1.6, 0.1, p)
    assert result.lambda_0 == pytest.approx(1.0)
    assert result.a == pytest.approx(0.0)
    assert result.eta_0 == pytest.approx(0.0)
    assert result.b == pytest.approx(0.0)
    assert result.stable is True


def test_no_equilibrium_returns_p_crit_boundary_equilibrium():
    """p beyond p_crit reports the real converged equilibrium at the p_crit boundary."""
    from sembra_vc.symbolic import evaluate_gradient

    alpha, Phi = 1.6, 0.0
    result = predict(alpha, Phi, 5.0)
    diag = result.diagnostics
    assert diag["branch"] == "no_equilibrium"
    assert result.stable is False
    assert diag["equilibrium_exists"] is False
    assert diag["requested_p_exceeds_p_crit"] is True

    # The returned state is a real converged equilibrium at p_crit.
    state = (result.lambda_0, result.a, result.eta_0, result.b)
    residual = np.linalg.norm(evaluate_gradient(alpha, Phi, diag["p_crit"], *state))
    assert residual < 1e-6

    # Full diagnostics are populated (non-None, finite) from the boundary state.
    assert diag["det_H"] is not None and np.isfinite(diag["det_H"])
    assert diag["t_1"] is not None and np.all(np.isfinite(diag["t_1"]))
    assert diag["t_2"] is not None and np.all(np.isfinite(diag["t_2"]))
    assert diag["E_max"] is not None and np.isfinite(diag["E_max"])

    # The message reports both p_crit and lambda_limit numerically.
    assert f"{diag['p_crit']:.4f}" in result.message
    assert f"{diag['lambda_limit']:.4f}" in result.message


def test_no_equilibrium_adaptive_margin_recovers():
    """A no_equilibrium query whose solve fails exactly at the singular fold
    recovers via the adaptive retreat and reports real diagnostics (DEV-018)."""
    result = predict(1.5, 0.105, 2.0)  # p >> p_crit; solve at p_crit fails, retreats
    diag = result.diagnostics
    assert diag["branch"] == "no_equilibrium"
    assert diag["newton_converged"] is True
    assert diag["boundary_margin"] is not None and diag["boundary_margin"] > 0.0
    assert diag["t_1"] is not None and np.all(np.isfinite(diag["t_1"]))
    assert np.isfinite(result.det_H)
    # The retreat margin is surfaced in the message for the user.
    assert "below p_crit" in result.message


def test_near_fold_returns_converged_equilibrium():
    """The near-fold band returns the Newton-converged 4D equilibrium, warned."""
    from sembra_vc.envelope import load_envelopes, predict_p_crit

    p_crit = predict_p_crit(load_envelopes()[0], 1.6, 0.0)
    result = predict(1.6, 0.0, p_crit)  # delta = 0 -> near_fold
    assert result.diagnostics["branch"] == "near_fold"
    assert result.diagnostics["near_fold_warning"] is True
    assert result.stable is True
    assert result.diagnostics["newton_converged"] is True
    assert result.diagnostics["equilibrium_exists"] is True
    # Converged equilibrium: full 4D residual is essentially zero.
    assert result.diagnostics["residual_norm"] < 1e-6


def test_near_fold_newton_failure_propagates(monkeypatch):
    """A forced Newton failure surfaces via equilibrium_exists/newton_converged."""
    from sembra_vc.envelope import load_envelopes, predict_p_crit

    class _FailedSolve:
        success = False
        x = np.array([1.5, 0.0, 0.0, 0.0])
        nfev = 7

    monkeypatch.setattr(wrapper_module, "root", lambda *a, **k: _FailedSolve())
    p_crit = predict_p_crit(load_envelopes()[0], 1.6, 0.0)
    result = predict(1.6, 0.0, p_crit)
    assert result.diagnostics["branch"] == "near_fold"
    assert result.stable is True  # branch semantic, independent of solve success
    assert result.diagnostics["newton_converged"] is False
    assert result.diagnostics["equilibrium_exists"] is False
    assert "WARNING" in result.message


def test_diagnostics_keys_present():
    """Required diagnostics keys are present on the reachable branches."""
    for result in (predict(1.6, 0.1, 0.0), predict(1.6, 0.0, 5.0)):
        assert _REQUIRED_DIAG_KEYS.issubset(result.diagnostics.keys())


def test_no_stable_region_branch():
    """No (alpha, Phi) in range yields p_crit <= 0, so this branch is unreachable here."""
    pytest.skip("No p_crit <= 0 region exists in the training range for this scenario.")


def test_stable_branch_end_to_end():
    """A stable query runs network -> inversion -> projection end-to-end."""
    import sembra_vc.wrapper as wrap
    from sembra_vc.envelope import load_envelopes, predict_p_crit
    from sembra_vc.network import load_network

    # The wrapper builds CPU input tensors, so use a CPU inference network.
    wrap._NETWORK = load_network(torch.device("cpu"))
    p_crit = predict_p_crit(load_envelopes()[0], 1.6, 0.1)
    result = predict(1.6, 0.1, 0.5 * p_crit)  # well below p_crit -> stable branch
    assert result.diagnostics["branch"] == "stable"
    assert isinstance(result.stable, bool)
    assert isinstance(result.diagnostics["projection_converged"], bool)
    assert np.isfinite(result.diagnostics["residual_norm"])


def test_stable_branch_on_gpu():
    """predict() stable branch works with the network on GPU (regression: DEV-008)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import sembra_vc.wrapper as wrap
    from sembra_vc.envelope import load_envelopes, predict_p_crit
    from sembra_vc.network import load_network

    wrap._NETWORK = load_network(torch.device("cuda"))
    try:
        p_crit = predict_p_crit(load_envelopes()[0], 1.6, 0.1)
        result = predict(1.6, 0.1, 0.5 * p_crit)
        assert result.diagnostics["branch"] == "stable"
        assert np.isfinite(result.diagnostics["residual_norm"])
    finally:
        # Restore a CPU inference network so later tests are unaffected.
        wrap._NETWORK = load_network(torch.device("cpu"))
