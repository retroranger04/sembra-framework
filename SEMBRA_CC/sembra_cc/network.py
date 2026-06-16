"""Surrogate MLP: (alpha, alpha_p, lambda_0) -> (p, eta_0, det_H).

Trained with a data MSE + LAMBDA_PHYS * physics-residual loss. The physics
residual uses a PyTorch reimplementation of the Section 4.3 integrand,
identical bit-for-bit to symbolic.py's _build_integrand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sembra_cc import TRAINED_MODELS_DIR
from sembra_cc.calibrations import LAMBDA_PHYS

NETWORK_PATH = TRAINED_MODELS_DIR / "network.pt"

_INPUT_FEATURES = ("alpha", "alpha_p", "lambda_0")
_OUTPUT_TARGETS = ("p", "eta_0", "det_H")

_N_GRID = 1001
_ALPHA_RANDOM_RANGE = (1.2, 2.0)
_ALPHA_P_RANDOM_RANGE = (0.0, 0.40)
_LAMBDA_0_RANDOM_RANGE = (1.0, 2.5)


def _build_trap_weights(device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    r_grid = torch.linspace(0.0, 1.0, _N_GRID, device=device, dtype=dtype)
    dr = 1.0 / (_N_GRID - 1)
    weights = torch.full((_N_GRID,), dr, device=device, dtype=dtype)
    weights[0] = dr / 2.0
    weights[-1] = dr / 2.0
    return r_grid, weights


def _torch_pi(alpha, alpha_p, p, lambda_0, eta_0, r_grid, weights):
    """Π evaluated by composite trapezoidal sum. Differentiable w.r.t. lambda_0, eta_0."""
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
    wecc_int = (ap * lambda_3 * lambda_3 / (2.0 * a)) * r
    wp_int = -0.5 * pp * (r * lambda_2) ** 2 * deta_dr

    integrand = u_int + wecc_int - wp_int
    return (integrand * weights).sum(dim=-1)


def _physics_residual_sq(alpha, alpha_p, p, lambda_0, eta_0, r_grid, weights):
    """Mean squared sum of partials dPi/dlambda_0 and dPi/deta_0."""
    l0 = lambda_0.detach().clone().requires_grad_(True)
    e0 = eta_0.detach().clone().requires_grad_(True) if not eta_0.requires_grad else eta_0
    # Re-do as fresh leaf if needed (autograd needs both leaves):
    if not e0.requires_grad:
        e0 = eta_0.detach().clone().requires_grad_(True)

    pi = _torch_pi(alpha, alpha_p, p, l0, e0, r_grid, weights)
    pi_sum = pi.sum()
    grad_l0, grad_e0 = torch.autograd.grad(
        pi_sum, [l0, e0], create_graph=True, retain_graph=True
    )
    return (grad_l0 * grad_l0 + grad_e0 * grad_e0).mean()


class SurrogateNetwork(nn.Module):
    """MLP with 3 -> 128 -> 128 -> 128 -> 3, SiLU activations."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))
        self.register_buffer("y_mean", torch.zeros(3))
        self.register_buffer("y_std", torch.ones(3))

    def forward(self, x_norm):
        return self.net(x_norm)

    def predict(self, x):
        """Denormalized inference. Input shape (..., 3) of (alpha, alpha_p, lambda_0)."""
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = self.net(x_norm)
        return y_norm * self.y_std + self.y_mean


def _set_scaling(model: SurrogateNetwork, x: np.ndarray, y: np.ndarray) -> None:
    model.x_mean.copy_(torch.tensor(x.mean(axis=0), dtype=torch.float32))
    model.x_std.copy_(torch.tensor(x.std(axis=0) + 1e-12, dtype=torch.float32))
    model.y_mean.copy_(torch.tensor(y.mean(axis=0), dtype=torch.float32))
    model.y_std.copy_(torch.tensor(y.std(axis=0) + 1e-12, dtype=torch.float32))


def _df_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[list(_INPUT_FEATURES)].to_numpy(dtype=np.float64)
    y = df[list(_OUTPUT_TARGETS)].to_numpy(dtype=np.float64)
    return x, y


def train_network(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    symbolic_funcs: dict,
    lambda_phys: float = LAMBDA_PHYS,
    seed: int = 42,
    max_epochs: int = 2000,
    patience: int = 100,
    batch_size: int = 256,
    save_path: Optional[Path] = NETWORK_PATH,
    verbose: bool = True,
) -> tuple[SurrogateNetwork, dict]:
    """Train the SurrogateNetwork on stable training data with a hybrid loss."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    dtype = torch.float32

    x_train_np, y_train_np = _df_to_arrays(train_df)
    x_val_np, y_val_np = _df_to_arrays(val_df)

    model = SurrogateNetwork().to(device=device, dtype=dtype)
    _set_scaling(model, x_train_np, y_train_np)

    x_train = torch.tensor(x_train_np, dtype=dtype, device=device)
    y_train = torch.tensor(y_train_np, dtype=dtype, device=device)
    x_val = torch.tensor(x_val_np, dtype=dtype, device=device)
    y_val = torch.tensor(y_val_np, dtype=dtype, device=device)

    # Normalized targets for the data loss (output of the network is normalized).
    y_train_norm = (y_train - model.y_mean) / model.y_std
    y_val_norm = (y_val - model.y_mean) / model.y_std
    x_train_norm = (x_train - model.x_mean) / model.x_std
    x_val_norm = (x_val - model.x_mean) / model.x_std

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    r_grid, weights = _build_trap_weights(device, dtype)

    n = x_train.shape[0]
    generator = torch.Generator().manual_seed(seed)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "l_data": [],
        "l_phys": [],
        "lambda_phys_times_l_phys": [],
        "lr": [],
    }

    final_l_data = float("nan")
    final_lambda_phys_l_phys = float("nan")

    rng_random = np.random.default_rng(seed + 1)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=generator, device=device)

        epoch_train_loss = 0.0
        epoch_l_data = 0.0
        epoch_l_phys = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x_batch_norm = x_train_norm[idx]
            y_batch_norm = y_train_norm[idx]
            x_batch = x_train[idx]

            pred_norm = model(x_batch_norm)
            l_data = ((pred_norm - y_batch_norm) ** 2).mean()

            # Denormalize predictions for the physics residual.
            pred = pred_norm * model.y_std + model.y_mean
            alpha_b = x_batch[:, 0]
            alpha_p_b = x_batch[:, 1]
            lambda_0_b = x_batch[:, 2]
            p_pred = pred[:, 0]
            eta_0_pred = pred[:, 1]

            l_phys_train = _physics_residual_sq(
                alpha_b, alpha_p_b, p_pred, lambda_0_b, eta_0_pred, r_grid, weights
            )

            # Random subset: sample uniformly in (alpha, alpha_p, lambda_0).
            b = x_batch.shape[0]
            rand = rng_random.uniform(
                low=[_ALPHA_RANDOM_RANGE[0], _ALPHA_P_RANDOM_RANGE[0], _LAMBDA_0_RANDOM_RANGE[0]],
                high=[_ALPHA_RANDOM_RANGE[1], _ALPHA_P_RANDOM_RANGE[1], _LAMBDA_0_RANDOM_RANGE[1]],
                size=(b, 3),
            ).astype(np.float32)
            x_rand = torch.tensor(rand, dtype=dtype, device=device)
            x_rand_norm = (x_rand - model.x_mean) / model.x_std
            pred_rand_norm = model(x_rand_norm)
            pred_rand = pred_rand_norm * model.y_std + model.y_mean
            l_phys_random = _physics_residual_sq(
                x_rand[:, 0],
                x_rand[:, 1],
                pred_rand[:, 0],
                x_rand[:, 2],
                pred_rand[:, 1],
                r_grid,
                weights,
            )

            l_phys = 0.5 * (l_phys_train + l_phys_random)
            loss = l_data + lambda_phys * l_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += float(loss.item())
            epoch_l_data += float(l_data.item())
            epoch_l_phys += float(l_phys.item())
            n_batches += 1

        scheduler.step()

        avg_train_loss = epoch_train_loss / max(n_batches, 1)
        avg_l_data = epoch_l_data / max(n_batches, 1)
        avg_l_phys = epoch_l_phys / max(n_batches, 1)
        avg_lambda_l_phys = lambda_phys * avg_l_phys

        # Validation loss (data-only).
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(x_val_norm)
            val_loss = float(((val_pred_norm - y_val_norm) ** 2).mean().item())

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["l_data"].append(avg_l_data)
        history["l_phys"].append(avg_l_phys)
        history["lambda_phys_times_l_phys"].append(avg_lambda_l_phys)
        history["lr"].append(float(scheduler.get_last_lr()[0]))

        final_l_data = avg_l_data
        final_lambda_phys_l_phys = avg_lambda_l_phys

        improved = val_loss < best_val_loss - 1e-12
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if verbose and (epoch % 20 == 0 or epoch == max_epochs - 1):
            print(
                f"epoch {epoch:4d}  train={avg_train_loss:.4e}  val={val_loss:.4e}  "
                f"l_data={avg_l_data:.4e}  l_phys={avg_l_phys:.4e}  "
                f"lambda*l_phys={avg_lambda_l_phys:.4e}  lr={history['lr'][-1]:.2e}"
            )

        if epochs_since_improvement >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best val {best_val_loss:.4e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history["best_val_loss"] = best_val_loss
    history["final_l_data"] = final_l_data
    history["final_lambda_phys_times_l_phys"] = final_lambda_phys_l_phys
    if final_lambda_phys_l_phys > 0:
        history["lambda_phys_ratio"] = final_l_data / final_lambda_phys_l_phys
    else:
        history["lambda_phys_ratio"] = float("inf")

    if save_path is not None:
        save_network(model, Path(save_path))

    return model, history


def save_network(model: SurrogateNetwork, path: Path = NETWORK_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "input_features": list(_INPUT_FEATURES),
        "output_targets": list(_OUTPUT_TARGETS),
    }
    torch.save(payload, path)


def load_network(path: str | Path = NETWORK_PATH) -> SurrogateNetwork:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = SurrogateNetwork()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def predict_from_lambda_0(
    model: SurrogateNetwork, alpha: float, alpha_p: float, lambda_0: float
) -> tuple[float, float, float]:
    x = torch.tensor([[alpha, alpha_p, lambda_0]], dtype=torch.float32)
    with torch.no_grad():
        y = model.predict(x)
    return float(y[0, 0]), float(y[0, 1]), float(y[0, 2])
