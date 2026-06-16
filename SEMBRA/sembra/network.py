"""Surrogate MLP for SEMBRA, trained with a hybrid data + physics loss.

Architecture and hyperparameters follow PRD Section 6.3 literally:

* MLP `(alpha, alpha_p, lambda_0)` → `(p, eta_0, det_H)` with three SiLU
  hidden layers of 128 units.
* Standard scaling on both inputs and outputs using training-set
  statistics; the scaling vectors travel with the checkpoint.
* Loss `L = L_data + lambda_phys * L_phys` where `L_data` is normalized-space
  MSE on (p, eta_0, det_H) and `L_phys` is the mean squared equilibrium
  residual computed in a differentiable PyTorch reimplementation of Section 4
  Pi, with autograd providing dPi/dlambda_0 and dPi/deta_0 at half-training,
  half-random sample points.
* Adam @ lr=1e-3 with default betas, batch size 256, cosine LR decay across
  the full 2000-epoch budget, early stop on validation loss with patience
  100, restore best weights.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sembra import TRAINED_MODELS_DIR
from sembra.symbolic import N_GRID

NETWORK_PATH = TRAINED_MODELS_DIR / "network.pt"

_INPUT_FEATURES = ("alpha", "alpha_p", "lambda_0")
_OUTPUT_TARGETS = ("p", "eta_0", "det_H")
_HIDDEN_DIM = 128
_LEARNING_RATE = 1e-3
_BATCH_SIZE = 256
_MAX_EPOCHS = 2000
_PATIENCE = 100
_LAMBDA_0_RANDOM_RANGE = (1.0, 2.4)
_ALPHA_RANDOM_RANGE = (1.2, 2.0)
_ALPHA_P_RANDOM_RANGE = (0.0, 0.72)

logger = logging.getLogger(__name__)


class SurrogateNetwork(nn.Module):
    """3 → 128 → 128 → 128 → 3 SiLU MLP with built-in I/O standard-scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(_INPUT_FEATURES), _HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(_HIDDEN_DIM, _HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(_HIDDEN_DIM, len(_OUTPUT_TARGETS)),
        )
        self.register_buffer("x_mean", torch.zeros(len(_INPUT_FEATURES)))
        self.register_buffer("x_std", torch.ones(len(_INPUT_FEATURES)))
        self.register_buffer("y_mean", torch.zeros(len(_OUTPUT_TARGETS)))
        self.register_buffer("y_std", torch.ones(len(_OUTPUT_TARGETS)))

    def set_normalization(
        self,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.x_mean.copy_(torch.as_tensor(x_mean, dtype=torch.float32))
        self.x_std.copy_(torch.as_tensor(x_std, dtype=torch.float32))
        self.y_mean.copy_(torch.as_tensor(y_mean, dtype=torch.float32))
        self.y_std.copy_(torch.as_tensor(y_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the normalized output (training-time call)."""
        return self.net(x)

    def predict(self, x_denorm: torch.Tensor) -> torch.Tensor:
        """Inference: accept denormalized input, return denormalized output."""
        x_n = (x_denorm - self.x_mean) / self.x_std
        y_n = self.net(x_n)
        return y_n * self.y_std + self.y_mean


def _torch_pi(
    alpha: torch.Tensor,
    alpha_p: torch.Tensor,
    p: torch.Tensor,
    lambda_0: torch.Tensor,
    eta_0: torch.Tensor,
    r_grid: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Pi computed by the same 1001-point trapezoidal rule."""
    r = r_grid.unsqueeze(0)
    a = alpha.unsqueeze(-1)
    ap = alpha_p.unsqueeze(-1)
    pp = p.unsqueeze(-1)
    l0 = lambda_0.unsqueeze(-1)
    e0 = eta_0.unsqueeze(-1)

    r_sq = r * r
    one_minus_r2 = 1.0 - r_sq
    lambda_2 = r_sq + l0 * one_minus_r2
    dl2_dr = 2.0 * r * (1.0 - l0)
    deta_dr = -2.0 * e0 * r
    inner = lambda_2 + r * dl2_dr
    lambda_1 = torch.sqrt(inner * inner + deta_dr * deta_dr)
    lambda_3 = 1.0 / (lambda_1 * lambda_2)

    u_int = (1.0 / a) * (
        torch.pow(lambda_1, a) + torch.pow(lambda_2, a) + torch.pow(lambda_3, a) - 3.0
    ) * r
    we_int = (ap / (2.0 * a * lambda_3 * lambda_3)) * r
    wp_int = -0.5 * pp * (r * lambda_2) ** 2 * deta_dr

    integrand = u_int - we_int - wp_int
    return (integrand * weights).sum(dim=-1)


def _physics_residual_loss(
    alpha: torch.Tensor,
    alpha_p: torch.Tensor,
    p: torch.Tensor,
    lambda_0: torch.Tensor,
    eta_0: torch.Tensor,
    r_grid: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    lambda_0_var = lambda_0.detach().clone().requires_grad_(True)
    pi_vals = _torch_pi(alpha, alpha_p, p, lambda_0_var, eta_0, r_grid, weights)
    pi_sum = pi_vals.sum()
    grad_lambda, grad_eta = torch.autograd.grad(
        pi_sum, [lambda_0_var, eta_0], create_graph=True, retain_graph=True
    )
    return (grad_lambda * grad_lambda + grad_eta * grad_eta).mean()


def train_network(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    symbolic_funcs: dict,
    lambda_phys: float,
    seed: int = 42,
    max_epochs: int = _MAX_EPOCHS,
    patience: int = _PATIENCE,
    batch_size: int = _BATCH_SIZE,
    save_path: Path | None = NETWORK_PATH,
) -> tuple[SurrogateNetwork, dict]:
    """Train the surrogate network with the hybrid data + physics loss.

    Parameters
    ----------
    train_df, val_df
        Training and validation DataFrames (output of
        :func:`sembra.data.make_train_val_split`).
    symbolic_funcs
        Output of :func:`sembra.symbolic.derive_symbolic_expressions`. Unused
        inside the training loop (the physics loss uses a differentiable
        PyTorch reimplementation of the integrand), but accepted for API
        parity with PRD Section 6.3.
    lambda_phys
        Weight on the physics-residual term.
    seed
        Seeds NumPy, Python ``random``, and PyTorch.
    max_epochs, patience, batch_size
        Training hyperparameters (PRD defaults).

    Returns
    -------
    tuple
        ``(trained_model, history)`` where ``history`` is a dict of per-epoch
        training/validation statistics, including ``l_data`` and
        ``lambda_phys_times_l_phys`` whose final-epoch ratio drives the
        LAMBDA_PHYS DEC entry.
    """
    _ = symbolic_funcs
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

    x_train = train_df[list(_INPUT_FEATURES)].to_numpy(dtype=np.float32)
    y_train = train_df[list(_OUTPUT_TARGETS)].to_numpy(dtype=np.float32)
    x_val = val_df[list(_INPUT_FEATURES)].to_numpy(dtype=np.float32)
    y_val = val_df[list(_OUTPUT_TARGETS)].to_numpy(dtype=np.float32)

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_val_n = (x_val - x_mean) / x_std
    y_val_n = (y_val - y_mean) / y_std

    x_train_t = torch.from_numpy(x_train_n)
    y_train_t = torch.from_numpy(y_train_n)
    x_val_t = torch.from_numpy(x_val_n)
    y_val_t = torch.from_numpy(y_val_n)

    model = SurrogateNetwork()
    model.set_normalization(x_mean, x_std, y_mean, y_std)

    r_grid = torch.linspace(0.0, 1.0, N_GRID, dtype=torch.float32)
    weights = torch.full((N_GRID,), 1.0 / (N_GRID - 1), dtype=torch.float32)
    weights[0] = 0.5 / (N_GRID - 1)
    weights[-1] = 0.5 / (N_GRID - 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    history: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "l_data": [],
        "l_phys": [],
        "lambda_phys_times_l_phys": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0
    alpha_lo, alpha_hi = _ALPHA_RANDOM_RANGE
    ap_lo, ap_hi = _ALPHA_P_RANDOM_RANGE
    l0_lo, l0_hi = _LAMBDA_0_RANDOM_RANGE
    x_mean_t = torch.as_tensor(x_mean, dtype=torch.float32)
    x_std_t = torch.as_tensor(x_std, dtype=torch.float32)
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32)
    y_std_t = torch.as_tensor(y_std, dtype=torch.float32)
    rng = np.random.default_rng(seed)

    for epoch in range(max_epochs):
        model.train()
        sum_l_data = 0.0
        sum_l_phys = 0.0
        n_batches = 0
        for x_batch_n, y_batch_n in loader:
            optimizer.zero_grad()
            y_pred_n = model(x_batch_n)
            l_data = F.mse_loss(y_pred_n, y_batch_n)

            y_pred = y_pred_n * y_std_t + y_mean_t
            x_batch = x_batch_n * x_std_t + x_mean_t
            p_pred = y_pred[:, 0]
            eta_pred = y_pred[:, 1]
            alpha_b = x_batch[:, 0]
            alpha_p_b = x_batch[:, 1]
            lambda_0_b = x_batch[:, 2]
            res_train = _physics_residual_loss(
                alpha_b, alpha_p_b, p_pred, lambda_0_b, eta_pred, r_grid, weights
            )

            b = x_batch.shape[0]
            rand_alpha = torch.from_numpy(
                rng.uniform(alpha_lo, alpha_hi, size=b).astype(np.float32)
            )
            rand_ap = torch.from_numpy(
                rng.uniform(ap_lo, ap_hi, size=b).astype(np.float32)
            )
            rand_l0 = torch.from_numpy(
                rng.uniform(l0_lo, l0_hi, size=b).astype(np.float32)
            )
            rand_x = torch.stack([rand_alpha, rand_ap, rand_l0], dim=1)
            rand_x_n = (rand_x - x_mean_t) / x_std_t
            rand_y_n = model(rand_x_n)
            rand_y = rand_y_n * y_std_t + y_mean_t
            rand_p = rand_y[:, 0]
            rand_eta = rand_y[:, 1]
            res_random = _physics_residual_loss(
                rand_alpha, rand_ap, rand_p, rand_l0, rand_eta, r_grid, weights
            )

            l_phys = 0.5 * (res_train + res_random)
            loss = l_data + lambda_phys * l_phys
            loss.backward()
            optimizer.step()

            sum_l_data += float(l_data.detach())
            sum_l_phys += float(l_phys.detach())
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred_n = model(x_val_t)
            val_loss = F.mse_loss(val_pred_n, y_val_t).item()

        epoch_l_data = sum_l_data / n_batches
        epoch_l_phys = sum_l_phys / n_batches
        weighted_l_phys = lambda_phys * epoch_l_phys
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_l_data + weighted_l_phys)
        history["val_loss"].append(val_loss)
        history["l_data"].append(epoch_l_data)
        history["l_phys"].append(epoch_l_phys)
        history["lambda_phys_times_l_phys"].append(weighted_l_phys)
        history["lr"].append(scheduler.get_last_lr()[0])

        improved = val_loss < best_val - 1e-12
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                "epoch=%d L_data=%.4e lambda*L_phys=%.4e val_loss=%.4e best_val=%.4e",
                epoch, epoch_l_data, weighted_l_phys, val_loss, best_val,
            )

        if epochs_without_improvement > patience:
            logger.info("Early stop at epoch %d (best val_loss=%.4e).", epoch, best_val)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None:
        save_network(model, save_path)
    return model, history


def save_network(model: SurrogateNetwork, path: Path = NETWORK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": _INPUT_FEATURES,
            "output_targets": _OUTPUT_TARGETS,
        },
        path,
    )


def load_network(path: str | Path = NETWORK_PATH) -> SurrogateNetwork:
    """Load a trained network from disk."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = SurrogateNetwork()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def predict_from_lambda_0(
    model: SurrogateNetwork,
    alpha: float,
    alpha_p: float,
    lambda_0: float,
) -> tuple[float, float, float]:
    """Run the network on a single ``(alpha, alpha_p, lambda_0)`` and denormalize the output."""
    with torch.no_grad():
        x = torch.tensor([[float(alpha), float(alpha_p), float(lambda_0)]], dtype=torch.float32)
        y = model.predict(x).squeeze(0)
    return float(y[0]), float(y[1]), float(y[2])
