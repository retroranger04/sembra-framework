"""Dataset loading, schema validation, and stable/holdout splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sembra_cc import DATA_PATH

_EXPECTED_COLUMNS = ["alpha", "alpha_p", "p", "lambda_0", "eta_0", "det_H"]

HELDOUT_SLICES = [(1.4, 0.04), (1.6, 0.16), (1.8, 0.24), (1.8, 0.36)]


def _validate_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != _EXPECTED_COLUMNS:
        raise ValueError(
            f"schema mismatch: expected columns {_EXPECTED_COLUMNS}, got {list(df.columns)}"
        )
    for col in _EXPECTED_COLUMNS:
        if df[col].dtype != np.float64:
            raise ValueError(
                f"column '{col}' has dtype {df[col].dtype}, expected float64"
            )
        nan_mask = df[col].isna()
        if nan_mask.any():
            indices = list(np.flatnonzero(nan_mask.to_numpy())[:10])
            raise ValueError(
                f"column '{col}' has {int(nan_mask.sum())} NaN values at row indices {indices}"
            )
        inf_mask = np.isinf(df[col].to_numpy())
        if inf_mask.any():
            indices = list(np.flatnonzero(inf_mask)[:10])
            raise ValueError(
                f"column '{col}' has {int(inf_mask.sum())} infinite values at row indices {indices}"
            )


def load_full_dataset() -> pd.DataFrame:
    """Read real_data.csv from DATA_PATH and validate schema."""
    df = pd.read_csv(DATA_PATH)
    _validate_schema(df)
    return df


def split_stable_unstable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition by sign of det_H. Stable: det_H > 0. Limit/unstable: det_H <= 0."""
    stable_df = df[df["det_H"] > 0].reset_index(drop=True)
    limit_df = df[df["det_H"] <= 0].reset_index(drop=True)
    return stable_df, limit_df


def _heldout_mask(df: pd.DataFrame, tol: float = 1e-9) -> np.ndarray:
    a = df["alpha"].to_numpy()
    ap = df["alpha_p"].to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for slice_alpha, slice_alpha_p in HELDOUT_SLICES:
        mask |= (np.abs(a - slice_alpha) < tol) & (np.abs(ap - slice_alpha_p) < tol)
    return mask


def make_train_val_split(
    stable_df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out four (alpha, alpha_p) slices, then split the rest 80/20."""
    mask = _heldout_mask(stable_df)
    heldout_df = stable_df[mask].reset_index(drop=True)
    remainder = stable_df[~mask].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(remainder))
    rng.shuffle(indices)
    n_train = int(0.8 * len(remainder))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_df = remainder.iloc[train_idx].reset_index(drop=True)
    val_df = remainder.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, heldout_df
