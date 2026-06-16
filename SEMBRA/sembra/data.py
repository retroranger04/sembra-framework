"""Data loading, validation, and train/validation splitting for SEMBRA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sembra import PROJECT_ROOT

EXPECTED_COLUMNS = ("alpha", "alpha_p", "p", "lambda_0", "eta_0", "det_H")
_SLICE_MATCH_TOL = 1e-6


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_full_dataset(path: str | Path = "Real_Data_VC.csv") -> pd.DataFrame:
    """Load the full SEMBRA dataset from CSV and validate its schema.

    Parameters
    ----------
    path
        Path to the CSV file. Relative paths resolve against the project root.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the six expected columns in the prescribed order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the column set or column order does not match Section 3.1.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Data file not found: {resolved}")
    df = pd.read_csv(resolved)
    actual = tuple(df.columns)
    if actual != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {resolved}: got {actual}, expected {EXPECTED_COLUMNS}."
        )
    return df


def split_stable_unstable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition rows by sign of ``det_H``.

    Returns
    -------
    tuple
        ``(stable_rows, limit_point_rows)`` where stable rows have
        ``det_H > 0`` and limit-point rows have ``det_H < 0``. Any rows with
        ``det_H == 0`` would raise; PRD Section 3.2 states all rows fall in
        one of the two strict-sign classes.
    """
    if "det_H" not in df.columns:
        raise ValueError("DataFrame must contain a 'det_H' column.")
    zero_mask = df["det_H"] == 0
    if zero_mask.any():
        raise ValueError(
            f"{int(zero_mask.sum())} rows have det_H == 0, which is unsupported by Section 3.2."
        )
    stable = df[df["det_H"] > 0].copy()
    limit = df[df["det_H"] < 0].copy()
    return stable, limit


def make_train_val_split(
    stable_df: pd.DataFrame,
    holdout_slices: list[tuple[float, float]],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stable rows into train/validation sets.

    Validation consists of every stable row from any ``(alpha, alpha_p)`` pair
    in ``holdout_slices``, plus 5% of the remaining rows sampled uniformly at
    random with the supplied seed.

    Parameters
    ----------
    stable_df
        DataFrame of stable rows (output of :func:`split_stable_unstable`).
    holdout_slices
        Non-empty list of ``(alpha, alpha_p)`` pairs to hold out entirely.
    seed
        Random seed for the additional 5% random hold-out.

    Returns
    -------
    tuple
        ``(train_df, val_df)``. The two DataFrames are disjoint and their
        union (by index) equals ``stable_df``.

    Raises
    ------
    ValueError
        If ``holdout_slices`` is empty.
    """
    if not holdout_slices:
        raise ValueError("holdout_slices must contain at least one (alpha, alpha_p) pair.")

    slice_mask = pd.Series(False, index=stable_df.index)
    for alpha_val, alpha_p_val in holdout_slices:
        match = (
            np.isclose(stable_df["alpha"], alpha_val, atol=_SLICE_MATCH_TOL)
            & np.isclose(stable_df["alpha_p"], alpha_p_val, atol=_SLICE_MATCH_TOL)
        )
        slice_mask |= match

    holdout_rows = stable_df[slice_mask]
    remaining = stable_df[~slice_mask]
    extra_val = remaining.sample(frac=0.05, random_state=seed)
    train_df = remaining.drop(extra_val.index)
    val_df = pd.concat([holdout_rows, extra_val]).sort_index()
    return train_df, val_df
