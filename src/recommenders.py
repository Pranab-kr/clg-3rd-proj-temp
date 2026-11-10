"""Baseline and explainable cold-start recommendation functions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CATEGORIES, TOP_K
from src.explanations import build_popularity_reason, build_recommendation_reason
from src.features import build_product_matrix, build_user_profile

_REQUIRED_PRODUCT_COLUMNS = {"product_id", "product_name", "brand", "category"}
_REQUIRED_INTERACTION_COLUMNS = {"user_id", "product_id"}
_OPTIONAL_NUMERIC_DEFAULTS = {
    "discount": 0.0,
    "rating": 0.0,
    "review_count": 0.0,
    "warranty_years": 0.0,
    "value_score": 0.0,
    "is_new_item": False,
    "is_new_brand": False,
}


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return int(top_k)


def _validate_products(products: pd.DataFrame) -> None:
    if not isinstance(products, pd.DataFrame):
        raise TypeError("products must be a pandas DataFrame")
    if products.empty:
        raise ValueError("products must contain at least one row")
    missing = sorted(_REQUIRED_PRODUCT_COLUMNS - set(products.columns))
    if missing:
        raise ValueError("products missing required columns: " + ", ".join(missing))
    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("product_id must contain non-null unique identifiers")


def _normalise_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError("interactions must be a pandas DataFrame")
    missing = sorted(_REQUIRED_INTERACTION_COLUMNS - set(interactions.columns))
    if missing:
        raise ValueError("interactions missing required columns: " + ", ".join(missing))
    return interactions


def _numeric_series(table: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in table:
        return pd.Series(default, index=table.index, dtype=float)
    raw_values = table[column]
    values = pd.to_numeric(raw_values, errors="coerce")
    malformed = raw_values.notna() & values.isna()
    non_finite = values.notna() & ~np.isfinite(values)
    if malformed.any() or non_finite.any():
        raise ValueError(f"{column} must contain numeric finite values or missing values")
    return values.fillna(default).astype(float)


def _min_max(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    minimum, maximum = values.min(), values.max()
    if maximum == minimum:
        return pd.Series(0.5, index=values.index, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _interaction_counts(products: pd.DataFrame, interactions: pd.DataFrame) -> pd.Series:
    if interactions.empty:
        return pd.Series(0.0, index=products.index)
    counts = interactions.groupby("product_id").size()
    return products["product_id"].map(counts).fillna(0).astype(float)


def _output_frame(products: pd.DataFrame, scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    top = scores.sort_values(
        ["score", "product_id"], ascending=[False, True], kind="mergesort"
    ).head(min(top_k, len(products))).copy()
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    columns = [
        "rank",
        "product_id",
        "product_name",
        "brand",
        "category",
        "score",
        "reason",
    ]
    optional = [column for column in ("is_new_item", "is_new_brand", "preference_match", "exposure_component") if column in top]
    return top[columns + optional].reset_index(drop=True)


def popular_recommendations(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Rank all products using historical activity and quality signals.

    Zero-history products remain eligible but receive no fabricated activity.
    Ties are resolved deterministically by product ID.
    """
    _validate_products(products)
    _validate_top_k(top_k)
    interactions = _normalise_interactions(interactions)

    frame = products.copy()
    activity = _interaction_counts(frame, interactions)
    activity_score = _min_max(activity)
    rating_score = _min_max(_numeric_series(frame, "rating"))
    review_score = _min_max(_numeric_series(frame, "review_count"))
    frame["score"] = 0.60 * activity_score + 0.25 * rating_score + 0.15 * review_score
    frame["reason"] = frame.apply(build_popularity_reason, axis=1)
    return _output_frame(frame, frame, top_k)


def _validate_preferences(user_preferences: Mapping[str, Any], products: pd.DataFrame) -> None:
    if not isinstance(user_preferences, Mapping):
        raise TypeError("user_preferences must be a dictionary-like mapping")
    category = user_preferences.get("preferred_category")
    if category not in set(CATEGORIES) or category not in set(products["category"].dropna()):
        raise ValueError("preferred_category must be a supported product category")


def _price_fit(products: pd.DataFrame, preferences: Mapping[str, Any]) -> pd.Series:
    try:
        low = float(preferences["budget_min"])
        high = float(preferences["budget_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("budget_min and budget_max must be numeric values") from exc
    if not np.isfinite([low, high]).all() or low < 0 or high <= low:
        raise ValueError("budget_min and budget_max must be finite and satisfy budget_min < budget_max")
    prices = _numeric_series(products, "price")
    distance = np.where(prices < low, (low - prices) / max(high - low, 1.0), 0.0)
    distance += np.where(prices > high, (prices - high) / max(high - low, 1.0), 0.0)
    return pd.Series(np.clip(1.0 - distance, 0.0, 1.0), index=products.index)


def cold_start_recommendations(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    user_preferences: dict,
    top_k: int = TOP_K,
    exposure_weight: float = 0.10,
) -> pd.DataFrame:
    """Recommend by content similarity, price fit, quality/value, and bounded exposure."""
    _validate_products(products)
    _validate_top_k(top_k)
    interactions = _normalise_interactions(interactions)
    _validate_preferences(user_preferences, products)
    if not np.isfinite(exposure_weight) or not 0 <= exposure_weight <= 0.25:
        raise ValueError("exposure_weight must be finite and between 0 and 0.25")

    matrix, encoder = build_product_matrix(products)
    profile = build_user_profile(user_preferences, encoder)
    similarity = cosine_similarity(matrix, profile).ravel()
    similarity = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
    category_match = products["category"].eq(user_preferences["preferred_category"]).astype(float)
    price_fit = _price_fit(products, user_preferences)
    value = _min_max(_numeric_series(products, "value_score"))
    rating = _min_max(_numeric_series(products, "rating"))
    warranty = _min_max(_numeric_series(products, "warranty_years"))
    discount = _min_max(_numeric_series(products, "discount"))
    is_new_item = products.get("is_new_item", pd.Series(False, index=products.index)).fillna(False).astype(bool)
    activity = _interaction_counts(products, interactions)
    zero_history = activity.eq(0)

    # Relevance dominates; the bounded exposure term is only available to
    # products with no observed history and is deliberately small.
    preference_match = (
        0.40 * similarity
        + 0.25 * category_match
        + 0.20 * price_fit
        + 0.15 * value
    )
    quality = 0.50 * rating + 0.25 * warranty + 0.25 * discount
    exposure = exposure_weight * (is_new_item & zero_history).astype(float)
    scores = products.copy()
    scores["preference_match"] = preference_match
    scores["exposure_component"] = exposure
    scores["score"] = 0.70 * preference_match + 0.20 * quality + exposure
    scores["reason"] = [
        build_recommendation_reason(row, match, bool(new_item))
        for (_, row), match, new_item in zip(products.iterrows(), preference_match, is_new_item)
    ]
    scores["is_new_item"] = is_new_item
    scores["is_new_brand"] = products.get("is_new_brand", pd.Series(False, index=products.index)).fillna(False).astype(bool)
    return _output_frame(products, scores, top_k)


def _user_preferences(
    user_id: str | None,
    users: pd.DataFrame,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
) -> dict[str, Any]:
    if user_id is not None and "user_id" in users and user_id in set(users["user_id"]):
        row = users.loc[users["user_id"] == user_id].iloc[0]
        return row.to_dict()
    user_rows = interactions.loc[interactions["user_id"] == user_id]
    if not user_rows.empty:
        product_map = products.set_index("product_id")
        seen = product_map.reindex(user_rows["product_id"].drop_duplicates()).dropna(subset=["category"])
        if not seen.empty:
            category = str(seen["category"].mode().iloc[0])
            category_products = products.loc[products["category"] == category]
            return {
                "preferred_category": category,
                "budget_min": float(category_products["price"].min()),
                "budget_max": float(category_products["price"].max() + 1),
                "preferred_features": str(seen["key_specifications"].iloc[0]),
            }
    category = str(products["category"].mode().iloc[0])
    category_products = products.loc[products["category"] == category]
    return {
        "preferred_category": category,
        "budget_min": float(category_products["price"].min()),
        "budget_max": float(category_products["price"].max() + 1),
        "preferred_features": str(category_products["key_specifications"].iloc[0]),
    }


def recommend_for_user(
    user_id: str | None,
    products: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
    top_k: int = TOP_K,
) -> dict[str, pd.DataFrame]:
    """Return baseline and proposed recommendations for a known or new user."""
    _validate_products(products)
    _validate_top_k(top_k)
    if not isinstance(users, pd.DataFrame):
        raise TypeError("users must be a pandas DataFrame")
    interactions = _normalise_interactions(interactions)
    preferences = _user_preferences(user_id, users, products, interactions)
    return {
        "baseline": popular_recommendations(products, interactions, top_k=top_k),
        "proposed": cold_start_recommendations(products, interactions, preferences, top_k=top_k),
    }
