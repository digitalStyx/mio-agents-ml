"""Rozpoznawanie odręcznej cyfry z obrazu 8×8 (format UCI optdigits)."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
CLASSES = list(range(10))
