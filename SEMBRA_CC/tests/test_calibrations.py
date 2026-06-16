"""Tests for sembra_cc.calibrations."""

from sembra_cc import calibrations


def test_constants_exact_values():
    assert calibrations.P_ZERO_THRESHOLD == 1e-6
    assert calibrations.GREY_ZONE_HALF_WIDTH == 0.005
    assert calibrations.LAMBDA_PHYS == 5.0
