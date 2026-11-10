import warnings

import pandas as pd
import pytest

from src.classification import (
    classification_report_table,
    compare_classifiers,
    make_classification_dataset,
)
from src.data_generation import generate_dataset


REQUIRED_CLASSIFIERS = {
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "K-Nearest Neighbors",
}
METRIC_COLUMNS = {"classifier", "accuracy", "precision", "recall", "f1"}


def test_interaction_dataset_returns_features_and_proxy_engagement_labels():
    """Breaks if interaction-derived proxy labels are absent or misaligned."""
    dataset = generate_dataset(seed=42, n_products=30, n_users=20, n_interactions=80)

    features, labels = make_classification_dataset(**dataset)

    assert len(features) == len(labels) == len(dataset["interactions"])
    assert labels.name == "engagement_proxy"
    assert set(labels.unique()) == {0, 1}
    assert {"preferred_category", "budget_min", "budget_max", "category", "price"} <= set(
        features.columns
    )
    assert "interaction_strength" not in features


def test_interaction_dataset_rejects_unknown_user_or_product_references():
    """Breaks if unmatched interaction IDs are silently converted to imputed features."""
    dataset = generate_dataset(seed=42, n_products=30, n_users=20, n_interactions=80)
    interactions = dataset["interactions"].copy()
    interactions.loc[0, "user_id"] = "unknown-user"

    with pytest.raises(ValueError, match="unknown user or product"):
        make_classification_dataset(dataset["users"], dataset["products"], interactions)


def test_classifier_comparison_returns_reproducible_bounded_metrics():
    """Breaks if a required classifier or deterministic bounded metric is missing."""
    dataset = generate_dataset(seed=42, n_products=30, n_users=20, n_interactions=80)
    features, labels = make_classification_dataset(**dataset)

    first = compare_classifiers(features, labels, seed=42)
    second = compare_classifiers(features, labels, seed=42)

    assert set(first["classifier"]) == REQUIRED_CLASSIFIERS
    assert METRIC_COLUMNS <= set(first.columns)
    assert first.loc[:, ["accuracy", "precision", "recall", "f1"]].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all()
    pd.testing.assert_frame_equal(
        first.round(10).reset_index(drop=True), second.round(10).reset_index(drop=True)
    )


def test_classifier_comparison_handles_string_features_without_dtype_warnings():
    """Breaks if string-feature detection emits a Pandas dtype deprecation warning."""
    dataset = generate_dataset(seed=42, n_products=30, n_users=20, n_interactions=80)
    features, labels = make_classification_dataset(**dataset)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compare_classifiers(features, labels, seed=42)

    assert len(result) == 4


def test_classifier_comparison_rejects_a_one_class_proxy_dataset():
    """Breaks if invalid one-class training data reaches a classifier silently."""
    features = pd.DataFrame({"price": [100.0, 200.0, 300.0, 400.0]})
    labels = pd.Series([1, 1, 1, 1], name="engagement_proxy")

    with pytest.raises(ValueError, match="both classes"):
        compare_classifiers(features, labels, seed=42)


def test_classification_report_table_exposes_standard_per_class_metrics():
    """Breaks if the human-readable report omits a standard classification metric."""
    report = classification_report_table(
        pd.Series([0, 0, 1, 1]), pd.Series([0, 1, 1, 1])
    )

    assert {"label", "precision", "recall", "f1", "support"} <= set(report.columns)
    positive = report.loc[report["label"] == "1"].iloc[0]
    assert positive["precision"] == pytest.approx(2 / 3)
    assert positive["recall"] == pytest.approx(1.0)
    assert positive["f1"] == pytest.approx(0.8)
