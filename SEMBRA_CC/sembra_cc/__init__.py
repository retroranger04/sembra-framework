from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
LOGS_DIR = PROJECT_ROOT / "logs"
PLOTS_DIR = LOGS_DIR / "plots"
DATA_PATH = PROJECT_ROOT / "Real_Data_CC.csv"
