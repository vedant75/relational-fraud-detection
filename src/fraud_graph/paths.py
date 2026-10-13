"""Single canonical path policy for scripts, notebooks, and training."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
FIGURES_DIR = ROOT / "artifacts" / "figures"


def ensure_output_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
