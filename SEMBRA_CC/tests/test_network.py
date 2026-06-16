"""Tests for sembra_cc.network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sembra_cc.network import (
    SurrogateNetwork,
    load_network,
    predict_from_lambda_0,
    save_network,
    train_network,
)
from sembra_cc.symbolic import derive_symbolic_expressions


def _fixture_df(n: int = 64, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "alpha": rng.uniform(1.2, 2.0, n),
            "alpha_p": rng.uniform(0.0, 0.4, n),
            "p": rng.uniform(0.05, 1.5, n),
            "lambda_0": rng.uniform(1.0, 2.0, n),
            "eta_0": rng.uniform(0.1, 1.0, n),
            "det_H": rng.uniform(0.01, 1.0, n),
        }
    )


def test_network_architecture():
    model = SurrogateNetwork()
    # input dim 3, output dim 3
    x = torch.randn(4, 3)
    y = model(x)
    assert y.shape == (4, 3)
    # check there are exactly four Linear layers
    linears = [m for m in model.net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4
    assert linears[0].in_features == 3 and linears[0].out_features == 128
    assert linears[1].in_features == 128 and linears[1].out_features == 128
    assert linears[2].in_features == 128 and linears[2].out_features == 128
    assert linears[3].in_features == 128 and linears[3].out_features == 3


def test_scaling_buffers_persist():
    model = SurrogateNetwork()
    with torch.no_grad():
        model.x_mean.copy_(torch.tensor([0.5, 0.6, 0.7]))
        model.x_std.copy_(torch.tensor([1.1, 1.2, 1.3]))
    sd = model.state_dict()
    assert "x_mean" in sd and "x_std" in sd
    model2 = SurrogateNetwork()
    model2.load_state_dict(sd)
    assert torch.allclose(model2.x_mean, model.x_mean)
    assert torch.allclose(model2.x_std, model.x_std)


def test_train_network_short_run(tmp_path):
    train = _fixture_df(64, seed=1)
    val = _fixture_df(32, seed=2)
    funcs = derive_symbolic_expressions()
    model, history = train_network(
        train,
        val,
        funcs,
        max_epochs=2,
        patience=10,
        batch_size=16,
        save_path=tmp_path / "test.pt",
        verbose=False,
    )
    assert len(history["epoch"]) >= 2
    assert (tmp_path / "test.pt").exists()


def test_save_load_roundtrip(tmp_path):
    train = _fixture_df(32, seed=3)
    val = _fixture_df(16, seed=4)
    funcs = derive_symbolic_expressions()
    model, _ = train_network(
        train,
        val,
        funcs,
        max_epochs=2,
        patience=5,
        batch_size=8,
        save_path=tmp_path / "rt.pt",
        verbose=False,
    )
    loaded = load_network(tmp_path / "rt.pt")
    x = torch.tensor([[1.5, 0.1, 1.2]], dtype=torch.float32)
    with torch.no_grad():
        y1 = model.predict(x)
        y2 = loaded.predict(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def test_predict_from_lambda_0_returns_floats():
    model = SurrogateNetwork()
    p, eta, det = predict_from_lambda_0(model, 1.5, 0.1, 1.2)
    assert isinstance(p, float)
    assert isinstance(eta, float)
    assert isinstance(det, float)
