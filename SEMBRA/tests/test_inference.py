"""Self-contained inference smoke test for the public SEMBRA API.

Exercises the user-facing ``sembra.wrapper.predict`` entry point with a valid
input from the middle of the supported range. Uses only the trained artifacts
shipped in ``trained_models/`` (loaded internally by the wrapper) — no dataset
or other external file is required.
"""

from __future__ import annotations

import math

from sembra.wrapper import predict


def test_predict_midrange_stable_equilibrium() -> None:
    # Mid-range, physically stable inputs: alpha=1.6, alpha_p=0.2, p=0.2.
    result = predict(1.6, 0.2, 0.2)

    # Returned object exposes the documented equilibrium fields.
    for field in ("lambda_0", "eta_0", "det_H", "stable"):
        assert hasattr(result, field), f"PredictionResult missing '{field}'"

    # All numeric outputs are finite (no NaN, no inf).
    for field in ("lambda_0", "eta_0", "det_H"):
        value = getattr(result, field)
        assert math.isfinite(value), f"{field} is not finite: {value!r}"

    # Physical sanity at these inputs.
    assert result.lambda_0 > 1.0, f"expected inflated equilibrium, got lambda_0={result.lambda_0}"
    assert result.det_H > 0.0, f"expected stable point (det_H>0), got det_H={result.det_H}"
    assert result.stable is True
