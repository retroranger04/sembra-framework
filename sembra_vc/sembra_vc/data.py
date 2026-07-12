"""Dataset loading utilities for the voltage-controlled scenario.

Light helpers for reading and splitting the training dataset. Used by the
notebook and by validation/analysis scripts. The CSV lives in the workspace
``data/`` directory (development context); it is not bundled inside the installed
package.
"""

from pathlib import Path

import pandas as pd

# The workspace data directory, resolved relative to this module:
# data.py -> sembra_vc (inner) -> sembra_vc (outer) -> SEMBRA_V2 -> data/.
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "training_data_vc.csv"


def load_full_dataset() -> pd.DataFrame:
    """Load the full voltage-controlled training dataset as a DataFrame."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {_DATA_PATH}. This helper expects the "
            "workspace 'data/' directory to be present."
        )
    return pd.read_csv(_DATA_PATH)


def split_stable_unstable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (stable, unstable) by the sign of the ``Hess`` column.

    Stable rows have ``Hess > 0``; the remainder (the single limit-point row per
    slice) are returned as the unstable set.
    """
    stable = df[df["Hess"] > 0].reset_index(drop=True)
    unstable = df[df["Hess"] <= 0].reset_index(drop=True)
    return stable, unstable
