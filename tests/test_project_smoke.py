"""End-to-end smoke test: documented commands -> artifacts -> demo helpers."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, TOP_K
from src.demo import (
    build_preferences,
    load_demo_data,
    load_saved_comparison,
    recommend_for_new_user,
)
from src.evaluation import evaluate_recommenders


def _run(script, *args):
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data_dir, results_dir = root / "data", root / "results"
    _run("generate_data.py", "--seed", "42", "--output-dir", str(data_dir))
    _run("run_experiments.py", "--seed", "42", "--output-dir", str(results_dir))
    return data_dir, results_dir


def test_generated_data_has_cold_start_scenarios(pipeline):
    data_dir, _ = pipeline
    data = load_demo_data(data_dir=data_dir)
    products, users, interactions = data["products"], data["users"], data["interactions"]
    new_ids = set(products.loc[products["is_new_item"], "product_id"])
    assert new_ids
    assert not new_ids & set(interactions["product_id"])
    assert users["is_new_user"].any()


def test_metadata_matches_generated_tables(pipeline):
    data_dir, results_dir = pipeline
    data = load_demo_data(data_dir=data_dir)
    meta = json.loads((results_dir / "experiment_metadata.json").read_text())
    assert meta["n_products"] == len(data["products"])
    assert meta["n_users"] == len(data["users"])
    assert meta["n_interactions"] == len(data["interactions"])
    assert "synthetic" in meta["data_source"].lower()


def test_saved_comparison_matches_fresh_evaluation(pipeline):
    data_dir, results_dir = pipeline
    data = load_demo_data(data_dir=data_dir)
    saved = load_saved_comparison(results_dir).set_index("method")
    fresh = evaluate_recommenders(
        data["products"], data["users"], data["interactions"], top_k=TOP_K
    ).set_index("method")
    for method in ("baseline", "proposed"):
        for column in ("precision_at_k", "recall_at_k", "catalog_coverage", "new_item_exposure"):
            assert saved.loc[method, column] == pytest.approx(fresh.loc[method, column])


def test_proposed_wins_cold_start_metrics_and_reports_relevance_honestly(pipeline):
    _, results_dir = pipeline
    saved = load_saved_comparison(results_dir).set_index("method")
    assert saved.loc["baseline", "new_item_exposure"] == 0
    assert saved.loc["proposed", "new_item_exposure"] > 0
    assert saved.loc["proposed", "catalog_coverage"] > saved.loc["baseline", "catalog_coverage"]
    assert saved.loc["proposed", "relevant_new_item_rate"] > saved.loc["baseline", "relevant_new_item_rate"]


def test_television_new_user_scenario_surfaces_new_item_only_in_proposed(pipeline):
    data_dir, _ = pipeline
    data = load_demo_data(data_dir=data_dir)
    prefs = build_preferences("Television", 300, 1800, ["4K", "HDR"])
    result = recommend_for_new_user(data, prefs)
    assert result["proposed"]["is_new_item"].any()
    assert not result["baseline"]["is_new_item"].any()
    new_rows = result["proposed"][result["proposed"]["is_new_item"]]
    assert new_rows["reason"].str.contains("no ", case=False).all()
    assert (result["proposed"]["category"] == "Television").any()


def test_results_are_reproducible_across_runs(pipeline, tmp_path):
    _, results_dir = pipeline
    _run("run_experiments.py", "--seed", "42", "--output-dir", str(tmp_path))
    for name in ("recommender_comparison.csv", "classifier_comparison.csv", "cluster_summary.csv"):
        assert (results_dir / name).read_text() == (tmp_path / name).read_text()
