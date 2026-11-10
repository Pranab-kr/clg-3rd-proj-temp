import numpy as np
import pytest

from src.clustering import fit_user_clusters, summarize_clusters
from src.data_generation import generate_users


def test_kmeans_returns_one_label_per_user_and_honors_cluster_count():
    users = generate_users(seed=42, n_users=12)

    model = fit_user_clusters(users, n_clusters=3, seed=42)

    assert model.n_clusters == 3
    assert model.labels_.shape == (len(users),)
    assert np.isfinite(model.cluster_centers_).all()


def test_cluster_summary_contains_ids_counts_and_interpretation():
    users = generate_users(seed=42, n_users=12)
    model = fit_user_clusters(users, n_clusters=3, seed=42)

    summary = summarize_clusters(users, model.labels_)

    assert {"cluster_id", "count"}.issubset(summary.columns)
    assert summary["count"].sum() == len(users)
    assert summary["cluster_id"].is_unique


def test_cluster_count_must_fit_number_of_users():
    users = generate_users(seed=42, n_users=3)

    with pytest.raises(ValueError, match="n_clusters"):
        fit_user_clusters(users, n_clusters=4, seed=42)


def test_cluster_count_must_fit_distinct_user_feature_rows():
    users = generate_users(seed=42, n_users=3)
    users["preferred_category"] = "Television"
    users["preferred_features"] = "4K|HDR"
    users["budget_min"] = 400.0
    users["budget_max"] = 900.0

    with pytest.raises(ValueError, match="distinct feature rows"):
        fit_user_clusters(users, n_clusters=2, seed=42)


def test_cluster_summary_rejects_non_finite_or_non_integral_labels():
    users = generate_users(seed=42, n_users=3)

    with pytest.raises(ValueError, match="finite"):
        summarize_clusters(users, np.array([0.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="integers"):
        summarize_clusters(users, np.array([0.0, 1.5, 1.0]))


def test_cluster_summary_accepts_integral_float_labels():
    users = generate_users(seed=42, n_users=3)

    summary = summarize_clusters(users, np.array([0.0, 1.0, 1.0]))

    assert summary["count"].sum() == len(users)


def test_cluster_summary_preserves_large_exact_integer_labels():
    users = generate_users(seed=42, n_users=3)
    large_label = np.int64(9_007_199_254_740_993)

    integer_summary = summarize_clusters(
        users, np.array([large_label, 1, 1], dtype=np.int64)
    )
    object_summary = summarize_clusters(
        users, np.array([int(large_label), 1, 1], dtype=object)
    )

    assert integer_summary["cluster_id"].tolist() == [1, int(large_label)]
    assert object_summary["cluster_id"].tolist() == [1, int(large_label)]


def test_cluster_summary_rejects_integer_labels_outside_int64_range():
    users = generate_users(seed=42, n_users=3)

    with pytest.raises(ValueError, match="supported range"):
        summarize_clusters(users, np.array([2**63, 1, 1], dtype=object))
    with pytest.raises(ValueError, match="supported range"):
        summarize_clusters(users, np.array([np.uint64(2**64 - 1), 1, 1]))
    with pytest.raises(ValueError, match="supported range"):
        summarize_clusters(users, np.array([float(2**63), 1.0, 1.0]))
