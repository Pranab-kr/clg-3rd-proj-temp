from pathlib import Path

from src.config import CATEGORIES, DATA_DIR, PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR, TOP_K


def test_configuration_defines_reproducible_project_defaults():
    assert isinstance(RANDOM_SEED, int)
    assert TOP_K == 5
    assert "Television" in CATEGORIES
    assert PROJECT_ROOT.is_dir()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RESULTS_DIR == PROJECT_ROOT / "results"
