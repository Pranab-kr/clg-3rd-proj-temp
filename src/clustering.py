"""User preference clustering and interpretable cluster summaries."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler


_USER_CATEGORICAL_FEATURES = ("preferred_category", "preferred_features")
_USER_NUMERIC_FEATURES = ("budget_min", "budget_max")


def _user_feature_frame(users: pd.DataFrame) -> pd.DataFrame:
    required = set(_USER_CATEGORICAL_FEATURES + _USER_NUMERIC_FEATURES)
    missing = sorted(required - set(users.columns))
    if missing:
        raise ValueError(f"users is missing required columns: {', '.join(missing)}")
    if users.empty:
        raise ValueError("users must contain at least one row")

    frame = pd.DataFrame(index=users.index)
    for column in _USER_CATEGORICAL_FEATURES:
        frame[column] = users[column].fillna("Unknown").map(_normalise)
    for column in _USER_NUMERIC_FEATURES:
        values = pd.to_numeric(users[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite, non-negative numeric values")
        frame[column] = values
    if (frame["budget_min"] >= frame["budget_max"]).any():
        raise ValueError("budget_min must be less than budget_max")
    return frame[list(_USER_CATEGORICAL_FEATURES) + list(_USER_NUMERIC_FEATURES)]


def _normalise(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip() for item in value]
    else:
        tokens = [token.strip() for token in str(value).split("|")]
    tokens = sorted({token for token in tokens if token})
    return "|".join(tokens) or "Unknown"


def fit_user_clusters(users: pd.DataFrame, n_clusters: int = 3, seed: int = 42) -> object:
    """Fit deterministic K-Means over user budgets and declared preferences."""
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)):
        raise ValueError("n_clusters must be an integer between 2 and the number of users")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if len(users) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of users")

    frame = _user_feature_frame(users)
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
                list(_USER_CATEGORICAL_FEATURES),
            ),
            ("numeric", StandardScaler(), list(_USER_NUMERIC_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    matrix = np.asarray(encoder.fit_transform(frame), dtype=float)
    distinct_rows = np.unique(matrix, axis=0).shape[0]
    if n_clusters > distinct_rows:
        raise ValueError(
            "n_clusters cannot exceed the number of distinct feature rows"
        )
    model = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init=10)
    model.fit(matrix)
    model.feature_encoder_ = encoder
    model.user_feature_columns_ = tuple(_USER_CATEGORICAL_FEATURES + _USER_NUMERIC_FEATURES)
    model.user_ids_ = users["user_id"].to_numpy(copy=True) if "user_id" in users else None
    return model


def _validate_cluster_labels(labels: np.ndarray, expected_length: int) -> np.ndarray:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or len(labels_array) != expected_length:
        raise ValueError("labels must contain exactly one value per user")

    # Validate each supported numeric representation without first converting
    # integer values to float.  That conversion loses precision above 2**53
    # and can make an out-of-range object value wrap when cast to int64.
    if labels_array.dtype.kind in "iu":
        for value in labels_array:
            _validate_integer_label(value)
    elif labels_array.dtype.kind == "f":
        for value in labels_array:
            _validate_float_label(value)
    elif labels_array.dtype.kind == "O":
        for value in labels_array:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError("labels must contain finite integers")
            if isinstance(value, Integral):
                _validate_integer_label(value)
            elif isinstance(value, Real):
                _validate_float_label(value)
            else:
                raise ValueError("labels must contain finite integers")
    else:
        raise ValueError("labels must contain finite integers")

    # Keep the original array and dtype so accepted integer labels are never
    # silently rewritten by validation.
    return labels_array


def _validate_integer_label(value: object) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("labels must contain finite integers")
    integer_value = int(value)
    int_info = np.iinfo(np.int64)
    if integer_value < int_info.min or integer_value > int_info.max:
        raise ValueError("labels must contain integers within the supported range")


def _validate_float_label(value: object) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("labels must contain finite integers")
    if not bool(np.isfinite(value)):
        raise ValueError("labels must contain finite integers")
    if not bool(value == np.floor(value)):
        raise ValueError("labels must contain integers")

    # int(value) is exact after the integral check.  Checking the Python int,
    # rather than a float representation of the bounds, also handles the
    # rounded float form of np.iinfo(np.int64).max correctly.
    integer_value = int(value)
    int_info = np.iinfo(np.int64)
    if integer_value < int_info.min or integer_value > int_info.max:
        raise ValueError("labels must contain integers within the supported range")
    if value != integer_value:
        raise ValueError("labels must contain safely representable integers")


def summarize_clusters(users: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return counts and preference aggregates for each assigned cluster."""
    if not isinstance(users, pd.DataFrame):
        raise TypeError("users must be a pandas DataFrame")
    labels_array = _validate_cluster_labels(labels, len(users))
    if len(users) == 0:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "count",
                "average_budget_min",
                "average_budget_max",
                "dominant_category",
                "common_features",
            ]
        )
    _user_feature_frame(users)

    assigned = users.copy()
    assigned["cluster_id"] = labels_array
    rows = []
    for cluster_id, group in assigned.groupby("cluster_id", sort=True):
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(group)),
                "average_budget_min": float(pd.to_numeric(group["budget_min"]).mean()),
                "average_budget_max": float(pd.to_numeric(group["budget_max"]).mean()),
                "dominant_category": _mode_or_unknown(group["preferred_category"]),
                "common_features": _mode_or_unknown(group["preferred_features"]),
            }
        )
    return pd.DataFrame(rows)


def _mode_or_unknown(values: pd.Series) -> str:
    modes = values.dropna().astype(str).sort_values().mode()
    return str(modes.iloc[0]) if not modes.empty else "Unknown"
