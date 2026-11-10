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
