"""SEMBRA: Surrogate model for dielectric elastomer membrane equilibrium and stability."""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Absolute path to the SEMBRA project root (the folder containing ``sembra/``).

All file I/O in the package — model checkpoints, dataset, log files, plot
artifacts — is resolved relative to this constant (DEC-014). The notebook
discovers it by walking up from its working directory until it finds a
``sembra/__init__.py`` and then ``from sembra import PROJECT_ROOT``.
"""

TRAINED_MODELS_DIR: Path = PROJECT_ROOT / "trained_models"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
PLOTS_DIR: Path = LOGS_DIR / "plots"
DATA_PATH: Path = PROJECT_ROOT / "Real_Data.csv"
