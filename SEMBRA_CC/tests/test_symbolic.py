"""Tests for sembra_cc.symbolic."""

from __future__ import annotations

import numpy as np
import pytest

from sembra_cc import symbolic
from sembra_cc.symbolic import derive_symbolic_expressions, SYMBOLIC_VERSION


def test_returned_keys():
    funcs = derive_symbolic_expressions()
    assert set(funcs.keys()) == {
        "Pi",
        "dPi_dlambda_0",
        "dPi_deta_0",
        "det_H_analytical",
    }


def test_scalar_inputs_return_float():
    funcs = derive_symbolic_expressions()
    for key in ["Pi", "dPi_dlambda_0", "dPi_deta_0", "det_H_analytical"]:
        result = funcs[key](1.5, 0.1, 0.3, 1.2, 0.4)
        assert isinstance(result, float)


def test_array_inputs_return_array():
    funcs = derive_symbolic_expressions()
    alpha = np.array([1.4, 1.6, 1.8])
    alpha_p = np.array([0.1, 0.2, 0.3])
    p = np.array([0.2, 0.3, 0.4])
    lambda_0 = np.array([1.1, 1.2, 1.3])
    eta_0 = np.array([0.3, 0.4, 0.5])
    for key in ["Pi", "dPi_dlambda_0", "dPi_deta_0", "det_H_analytical"]:
        result = funcs[key](alpha, alpha_p, p, lambda_0, eta_0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


def test_cache_roundtrip(tmp_path, monkeypatch):
    # Force rebuild and ensure subsequent call hits cache deterministically.
    funcs1 = derive_symbolic_expressions(use_cache=False)
    funcs2 = derive_symbolic_expressions(use_cache=True)
    val1 = funcs1["det_H_analytical"](1.5, 0.1, 0.3, 1.2, 0.4)
    val2 = funcs2["det_H_analytical"](1.5, 0.1, 0.3, 1.2, 0.4)
    assert np.isclose(val1, val2, rtol=1e-12, atol=1e-12)


def test_cache_version_mismatch_rebuilds(monkeypatch):
    import cloudpickle

    cache_path = symbolic._CACHE_PATH
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        cloudpickle.dump({"version": "0.0.0-bogus", "callables": {}}, fh)
    funcs = derive_symbolic_expressions(use_cache=True)
    assert "Pi" in funcs
    assert callable(funcs["Pi"])
    # Confirm it now contains a valid version on disk.
    with open(cache_path, "rb") as fh:
        payload = cloudpickle.load(fh)
    assert payload["version"] == SYMBOLIC_VERSION
