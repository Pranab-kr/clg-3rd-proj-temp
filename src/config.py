"""Shared project configuration and reproducibility defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_SEED = 42
TOP_K = 5
CATEGORIES = ("Television", "Phone", "Laptop")
