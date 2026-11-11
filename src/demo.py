"""UI-independent helpers behind the Streamlit demo (kept out of app.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import DATA_DIR, RANDOM_SEED, RESULTS_DIR, TOP_K
from src.data_generation import generate_dataset
from src.data_validation import (
    validate_interactions,
    validate_products,
    validate_users,
)
from src.recommenders import (
    cold_start_recommendations,
    popular_recommendations,
    recommend_for_user,
)

SYNTHETIC_NOTICE = (
    "All products, users, and interactions in this demo are synthetic, generated "
    "locally from a fixed random seed. They are not real customer data and do not "
    "come from any marketplace."
)

_TABLES = ("products", "users", "interactions")


def load_demo_data(
    seed: int = RANDOM_SEED, data_dir: str | Path | None = DATA_DIR
) -> dict[str, pd.DataFrame]:
    """Load the saved synthetic CSV snapshot, or regenerate it from the seed."""
    if data_dir is not None:
        folder = Path(data_dir)
        paths = {name: folder / f"{name}.csv" for name in _TABLES}
        if all(path.exists() for path in paths.values()):
            data = {name: pd.read_csv(path) for name, path in paths.items()}
            errors = (
                validate_products(data["products"])
                + validate_users(data["users"])
                + validate_interactions(data["interactions"])
            )
            if not errors:
                return data
    return generate_dataset(seed)


def feature_options(products: pd.DataFrame, category: str) -> list[str]:
    """Sorted individual feature tokens available for a category."""
    specs = products.loc[products["category"] == category, "key_specifications"].dropna()
    tokens = {token.strip() for spec in specs for token in str(spec).split("|") if token.strip()}
    return sorted(tokens)


def price_bounds(products: pd.DataFrame, category: str) -> tuple[float, float]:
    """Observed (min, max) price for a category, for sensible budget sliders."""
    prices = products.loc[products["category"] == category, "price"]
    return float(prices.min()), float(prices.max())


def build_preferences(
    category: str,
    budget_min: float,
    budget_max: float,
    features: Iterable[str] = (),
) -> dict[str, Any]:
    """Assemble the preference mapping the recommenders expect."""
    return {
        "preferred_category": category,
        "budget_min": float(budget_min),
        "budget_max": float(budget_max),
        "preferred_features": "|".join(sorted(features)) or "Unknown",
    }


def recommend_for_preferences(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    preferences: dict[str, Any],
    top_k: int = TOP_K,
    exposure_weight: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Baseline and proposed tables for a preference-only (new-user) request."""
    return {
        "baseline": popular_recommendations(products, interactions, top_k=top_k),
        "proposed": cold_start_recommendations(
            products,
            interactions,
            preferences,
            top_k=top_k,
            exposure_weight=exposure_weight,
        ),
    }


def recommend_for_new_user(
    data: dict[str, pd.DataFrame],
    preferences: dict[str, Any],
    top_k: int = TOP_K,
    exposure_weight: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Cold-start scenario: the user has no history of their own.

    The baseline still ranks by the catalog's historical popularity (what a
    marketplace would show anyone); the proposed method needs only the stated
    preferences and product metadata.
    """
    return recommend_for_preferences(
        data["products"],
        data["interactions"],
        preferences,
        top_k=top_k,
        exposure_weight=exposure_weight,
    )


def recommend_for_known_user(
    user_id: str, data: dict[str, pd.DataFrame], top_k: int = TOP_K
) -> dict[str, pd.DataFrame]:
    """Baseline and proposed tables for a user already in the synthetic table."""
    return recommend_for_user(
        user_id, data["products"], data["users"], data["interactions"], top_k=top_k
    )


def load_saved_comparison(results_dir: str | Path = RESULTS_DIR) -> pd.DataFrame | None:
    """Saved baseline-vs-proposed metrics from run_experiments.py, if present."""
    path = Path(results_dir) / "recommender_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
