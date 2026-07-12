"""Neural-network surrogate for the voltage-controlled scenario.

Defines :class:`SembraNet`, the feedforward MLP that maps the conditioning
inputs ``(alpha, Phi, lambda_0)`` to the Ritz outputs ``(p, a, eta_0, b)``, plus
utilities to load trained weights and to save weights during training.

The network's only job at inference is to produce a good initial guess for the
projection layer (added in a later prompt); it does not itself enforce
equilibrium. It has no dependency on the symbolic core or the GP envelopes --
those are integrated in the wrapper, not here.

The architecture is fixed: ``3 -> 128 -> 128 -> 128 -> 4`` with SiLU (swish)
activations between the hidden layers and a linear output layer. No batch
normalization, no dropout.
"""

from pathlib import Path

import torch
from torch import nn

# Architecture constants describing the fixed MLP shape. These are structural
# facts of the model (not tunable calibrations), so they live with the module
# that defines the architecture.
_INPUT_DIM = 3
_HIDDEN_DIM = 128
_HIDDEN_LAYERS = 3
_OUTPUT_DIM = 4

# Location of the shipped weights inside the package.
_WEIGHTS_PATH = Path(__file__).resolve().parent / "trained_models" / "network.pt"

# Cache of loaded networks, keyed by resolved device string. Populated lazily by
# ``load_network`` so repeated calls return the same object.
_NETWORK_CACHE: dict[str, "SembraNet"] = {}


class SembraNet(nn.Module):
    """The SEMBRA MLP.

    Input:  ``(alpha, Phi, lambda_0)`` -- shape ``(batch, 3)``.
    Output: ``(p, a, eta_0, b)``       -- shape ``(batch, 4)``.

    Three hidden layers of width 128 with SiLU activations between them and a
    linear output layer.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(_INPUT_DIM, _HIDDEN_DIM), nn.SiLU()]
        for _ in range(_HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(_HIDDEN_DIM, _OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map inputs ``(batch, 3)`` to outputs ``(batch, 4)``."""
        return self.net(x)


def _auto_device(device: torch.device | None) -> torch.device:
    """Resolve the inference device: use the given one, else GPU if available."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_network(device: torch.device | None = None) -> SembraNet:
    """Load the trained network from the package's ``trained_models/network.pt``.

    The device is auto-detected when not supplied (GPU if available, else CPU).
    The returned network is in ``eval`` mode. Results are cached per device, so
    repeated calls with the same device return the same object.
    """
    resolved = _auto_device(device)
    key = str(resolved)
    if key in _NETWORK_CACHE:
        return _NETWORK_CACHE[key]

    if not _WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {_WEIGHTS_PATH}. Train the network "
            "first (see the training pipeline) before calling load_network()."
        )

    model = SembraNet()
    state_dict = torch.load(_WEIGHTS_PATH, map_location=resolved)
    model.load_state_dict(state_dict)
    model.to(resolved)
    model.eval()

    _NETWORK_CACHE[key] = model
    return model


def save_network(model: SembraNet, path: Path) -> None:
    """Save ``model``'s ``state_dict`` to ``path`` (parent dirs created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
