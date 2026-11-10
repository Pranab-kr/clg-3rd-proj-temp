"""Stable product and preference feature preparation for recommendations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = ("category", "key_specifications")
NUMERIC_FEATURES = (
    "price",
    "discount",
    "rating",
    "warranty_years",
    "value_score",
    "brand_age_years",
    "product_age_years",
    "review_count",
)

_DEFAULT_NUMERIC_VALUES = {
    "price": 0.0,
    "discount": 0.0,
    "rating": 3.5,
    "warranty_years": 1.0,
    "value_score": 0.5,
    "brand_age_years": 0.0,
    "product_age_years": 0.0,
    "review_count": 0.0,
}


def _as_feature_frame(products: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(products, pd.DataFrame):
        raise TypeError("products must be a pandas DataFrame")
    if products.empty:
        raise ValueError("products must contain at least one row")

    frame = pd.DataFrame(index=products.index)
    for column in CATEGORICAL_FEATURES:
        values = (
            products[column]
            if column in products
            else pd.Series("Unknown", index=products.index)
        )
        frame[column] = values.fillna("Unknown").map(_normalise_feature_value)

    for column in NUMERIC_FEATURES:
        if column not in products:
            frame[column] = _DEFAULT_NUMERIC_VALUES[column]
            continue
        raw_values = products[column]
        values = pd.to_numeric(raw_values, errors="coerce")
        malformed = raw_values.notna() & values.isna()
        non_finite = values.notna() & ~np.isfinite(values)
        if malformed.any() or non_finite.any():
            raise ValueError(f"{column} must contain numeric finite values or missing values")
        frame[column] = values.fillna(_DEFAULT_NUMERIC_VALUES[column])

    return frame[list(CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES)]


def _normalise_feature_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip() for item in value]
    else:
        tokens = [token.strip() for token in str(value).split("|")]
    tokens = sorted({token for token in tokens if token})
    return "|".join(tokens) or "Unknown"


def build_product_matrix(products: pd.DataFrame) -> tuple[np.ndarray, object]:
    """Encode product metadata into a finite, reusable numeric matrix.

    Categorical fields use one-hot encoding and numeric fields are standardized
    together. The returned fitted transformer is accepted by
    :func:`build_user_profile` for a zero-interaction new user.
    """
    frame = _as_feature_frame(products)
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    matrix = np.asarray(encoder.fit_transform(frame), dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("product features must produce finite values")

    encoder.feature_input_columns_ = tuple(CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    encoder.numeric_defaults_ = dict(_DEFAULT_NUMERIC_VALUES)
    return matrix, encoder


def _parse_budget(user_preferences: Mapping[str, Any]) -> tuple[float, float]:
    try:
        budget_min = float(user_preferences["budget_min"])
        budget_max = float(user_preferences["budget_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("budget_min and budget_max must be numeric values") from exc

    if not np.isfinite([budget_min, budget_max]).all() or budget_min < 0 or budget_max < 0:
        raise ValueError("budget_min and budget_max must be finite, non-negative values")
    if budget_min >= budget_max:
        raise ValueError("budget_min and budget_max must satisfy budget_min < budget_max")
    return budget_min, budget_max


def build_user_profile(user_preferences: dict, feature_encoder: object) -> np.ndarray:
    """Build a product-space vector for a new user from supplied preferences."""
    if not isinstance(user_preferences, Mapping):
        raise TypeError("user_preferences must be a dictionary-like mapping")
    if not hasattr(feature_encoder, "transform"):
        raise TypeError("feature_encoder must be a fitted transformer")

    budget_min, budget_max = _parse_budget(user_preferences)
    category = _normalise_feature_value(user_preferences.get("preferred_category", "Unknown"))
    preferred_features = _normalise_feature_value(
        user_preferences.get("preferred_features", "Unknown")
    )
    defaults = getattr(feature_encoder, "numeric_defaults_", _DEFAULT_NUMERIC_VALUES)
    profile = {
        "category": category,
        "key_specifications": preferred_features,
        **defaults,
    }
    profile["price"] = (budget_min + budget_max) / 2
    frame = pd.DataFrame([profile], columns=list(CATEGORICAL_FEATURES) + list(NUMERIC_FEATURES))
    vector = np.asarray(feature_encoder.transform(frame), dtype=float)
    if vector.ndim == 1:
        vector = vector.reshape(1, -1)
    if not np.isfinite(vector).all():
        raise ValueError("user profile features must be finite")
    return vector
