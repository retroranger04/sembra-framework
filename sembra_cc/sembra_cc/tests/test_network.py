"""Tests for the charge-controlled network module.

Exercises the fixed MLP architecture: a forward pass shape check and the exact
parameter count. The trained-weights load path is skipped until weights are
produced in the training prompt.
"""

import torch

from sembra_cc.network import SembraNet, load_network

# Expected parameter count for the 3 -> 128 -> 128 -> 128 -> 4 MLP:
#   (3*128 + 128) + (128*128 + 128) + (128*128 + 128) + (128*4 + 4)
EXPECTED_PARAM_COUNT = 3 * 128 + 128 + 128 * 128 + 128 + 128 * 128 + 128 + 128 * 4 + 4


def test_forward_shape():
    """A single (alpha, Phi, lambda_0) example maps to a (1, 4) output."""
    net = SembraNet()
    out = net(torch.tensor([[1.6, 0.1, 0.7]]))
    assert out.shape == (1, 4)


def test_parameter_count():
    """The architecture has exactly the parameter count the spec predicts."""
    net = SembraNet()
    total = sum(p.numel() for p in net.parameters())
    assert EXPECTED_PARAM_COUNT == 34052
    assert total == EXPECTED_PARAM_COUNT


def test_load_network_loads_trained_weights():
    """The shipped weights load cleanly; network is in eval mode; forward works."""
    net = load_network(torch.device("cpu"))
    assert isinstance(net, SembraNet)
    assert net.training is False
    out = net(torch.tensor([[1.6, 0.1, 1.3]], dtype=torch.float32))
    assert out.shape == (1, 4)
