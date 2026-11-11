import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.run_experiments import run_experiments

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_run_experiments_writes_tables_charts_and_metadata(tmp_path):
    outputs = run_experiments(seed=42, output_dir=tmp_path)

    for name in (
        "recommender_comparison.csv",
        "classifier_comparison.csv",
        "cluster_summary.csv",
        "experiment_metadata.json",
        "recommender_comparison.png",
        "classifier_comparison.png",
    ):
        assert (tmp_path / name).is_file(), name
    assert set(outputs) >= {"recommender_comparison", "classifier_comparison", "cluster_summary"}

    metadata = json.loads((tmp_path / "experiment_metadata.json").read_text())
    assert metadata["seed"] == 42
    assert metadata["top_k"] == 5
    assert "synthetic" in metadata["data_source"].lower()
    assert "proxy" in metadata["classification_target"].lower()
    assert isinstance(metadata["silhouette_score"], float)

    table = pd.read_csv(tmp_path / "recommender_comparison.csv")
    assert list(table["method"]) == ["baseline", "proposed"]


def test_run_experiments_is_reproducible(tmp_path):
    run_experiments(seed=42, output_dir=tmp_path / "a")
    run_experiments(seed=42, output_dir=tmp_path / "b")
    for name in ("recommender_comparison.csv", "classifier_comparison.csv", "cluster_summary.csv"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_cli_runs_from_repository_root(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/run_experiments.py", "--seed", "42", "--output-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "recommender_comparison.csv").is_file()
