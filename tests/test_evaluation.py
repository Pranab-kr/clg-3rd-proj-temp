import pandas as pd
import pytest

from src.data_generation import generate_dataset
from src.evaluation import (
    catalog_coverage,
    evaluate_recommenders,
    new_item_exposure,
    precision_at_k,
    recall_at_k,
    split_holdout,
)


def test_precision_and_recall_at_k_exact_values():
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(recommended, relevant, 2) == pytest.approx(0.5)
    assert precision_at_k(recommended, relevant, 4) == pytest.approx(0.5)
    assert recall_at_k(recommended, relevant, 2) == pytest.approx(1 / 3)
    assert recall_at_k(recommended, relevant, 4) == pytest.approx(2 / 3)


def test_precision_divides_by_k_when_list_is_short():
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)


def test_zero_denominators_and_invalid_k():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0
    assert precision_at_k([], {"a"}, 3) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, -1)


def test_catalog_coverage_is_bounded_and_counts_unique_items():
    lists = [["a", "b"], ["b", "c"]]
    assert catalog_coverage(lists, 6) == pytest.approx(0.5)
    assert catalog_coverage([], 6) == 0.0
    assert 0.0 <= catalog_coverage([["a", "b", "c"]], 3) <= 1.0
    with pytest.raises(ValueError):
        catalog_coverage(lists, 0)


def test_new_item_exposure_is_share_of_recommendation_slots():
    lists = [["a", "n1"], ["n1", "n2"]]
    assert new_item_exposure(lists, {"n1", "n2"}) == pytest.approx(0.75)
    assert new_item_exposure(lists, set()) == 0.0
    assert new_item_exposure([], {"n1"}) == 0.0


def test_split_holdout_is_deterministic_and_leaves_no_leakage():
    data = generate_dataset(seed=42)
    train_a, held_a = split_holdout(data["interactions"])
    train_b, held_b = split_holdout(data["interactions"])
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(held_a, held_b)
    assert len(train_a) + len(held_a) == len(data["interactions"])
    assert (held_a["interaction_strength"] >= 2).all()
    # Exactly one held-out positive per user, and only for users who keep
    # at least one other interaction in training.
    assert held_a["user_id"].is_unique
    assert set(held_a["user_id"]) <= set(train_a["user_id"])


def test_evaluation_compares_both_methods_with_expected_columns():
    data = generate_dataset(seed=42)
    table = evaluate_recommenders(
        data["products"], data["users"], data["interactions"], top_k=5
    )
    assert list(table["method"]) == ["baseline", "proposed"]
    expected = {
        "precision_at_k",
        "recall_at_k",
        "catalog_coverage",
        "new_item_exposure",
        "relevant_new_item_rate",
        "avg_rating",
        "avg_value_score",
        "n_users",
        "k",
    }
    assert expected <= set(table.columns)
    for column in ("precision_at_k", "recall_at_k", "catalog_coverage", "new_item_exposure", "relevant_new_item_rate"):
        assert table[column].between(0.0, 1.0).all()
    assert (table["k"] == 5).all()
    # Same users and K for both methods.
    assert table["n_users"].nunique() == 1 and table["n_users"].iloc[0] > 0


def test_baseline_never_exposes_zero_history_new_items_but_proposed_can():
    data = generate_dataset(seed=42)
    table = evaluate_recommenders(
        data["products"], data["users"], data["interactions"], top_k=5
    ).set_index("method")
    assert table.loc["baseline", "new_item_exposure"] == 0.0
    assert table.loc["proposed", "new_item_exposure"] > 0.0
    assert table.loc["proposed", "relevant_new_item_rate"] > 0.0


def test_evaluation_is_reproducible_for_same_seed():
    first = generate_dataset(seed=7)
    second = generate_dataset(seed=7)
    a = evaluate_recommenders(first["products"], first["users"], first["interactions"])
    b = evaluate_recommenders(second["products"], second["users"], second["interactions"])
    pd.testing.assert_frame_equal(a, b)


def test_evaluation_rejects_invalid_inputs():
    data = generate_dataset(seed=42)
    with pytest.raises(ValueError):
        evaluate_recommenders(data["products"], data["users"], data["interactions"], top_k=0)
    with pytest.raises(ValueError):
        evaluate_recommenders(
            data["products"], data["users"], data["interactions"].head(0)
        )
