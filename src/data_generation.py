"""Reproducible synthetic electronics data for the project demonstration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import CATEGORIES
from src.data_validation import (
    validate_interactions,
    validate_products,
    validate_users,
)


_CATEGORY_SPECS = {
    "Television": {
        "brands": ("VisionArc", "HomeBright", "NovaView", "ClearPeak"),
        "features": ("4K", "HDR", "Smart TV", "120Hz"),
        "price_range": (280.0, 1800.0),
    },
    "Phone": {
        "brands": ("VoltMobile", "PocketPro", "NovaCell", "BrightTech"),
        "features": ("5G", "OLED", "Fast Charge", "Dual Camera"),
        "price_range": (180.0, 1400.0),
    },
    "Laptop": {
        "brands": ("Northstar", "WorkLite", "NovaCompute", "BrightBook"),
        "features": ("16GB RAM", "SSD", "Backlit Keyboard", "Long Battery"),
        "price_range": (420.0, 2400.0),
    },
}
_EVENT_STRENGTH = {"view": 1.0, "wishlist": 2.0, "cart": 3.0, "purchase": 4.0}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _feature_string(features: tuple[str, ...], rng: np.random.Generator) -> str:
    selected = rng.choice(features, size=2, replace=False)
    return "|".join(sorted(selected.tolist()))


def generate_products(seed: int, n_products: int = 120) -> pd.DataFrame:
    """Generate a deterministic product catalog with reserved cold-start items."""
    if n_products < len(CATEGORIES):
        raise ValueError(f"n_products must be at least {len(CATEGORIES)}")

    rng = _rng(seed)
    rows: list[dict[str, Any]] = []
    cold_start_count = min(max(3, n_products // 12), n_products - 1)
    for index in range(n_products):
        category = CATEGORIES[index % len(CATEGORIES)]
        spec = _CATEGORY_SPECS[category]
        min_price, max_price = spec["price_range"]
        is_new_item = index < cold_start_count
        is_new_brand = is_new_item
        brand = f"NovaLaunch{index + 1}" if is_new_brand else str(rng.choice(spec["brands"]))
        price = round(float(rng.uniform(min_price, max_price)), 2)
        discount = round(float(rng.uniform(0.05, 0.35) if is_new_item else rng.uniform(0, 0.25)), 3)
        rating = round(float(rng.uniform(3.8, 4.9) if is_new_item else rng.uniform(3.2, 4.8)), 2)
        rows.append(
            {
                "product_id": f"p{index + 1:04d}",
                "product_name": f"{brand} {category} {index + 1:03d}",
                "category": category,
                "brand": brand,
                "brand_age_years": int(rng.integers(1, 3) if is_new_brand else rng.integers(5, 31)),
                "product_age_years": int(rng.integers(0, 2) if is_new_item else rng.integers(1, 6)),
                "price": price,
                "discount": discount,
                "rating": rating,
                "review_count": int(rng.integers(0, 8) if is_new_item else rng.integers(20, 5000)),
                "warranty_years": int(rng.integers(2, 5) if is_new_item else rng.integers(1, 4)),
                "key_specifications": _feature_string(spec["features"], rng),
                "value_score": round(float(np.clip((max_price - price) / (max_price - min_price) * 0.55 + discount * 0.8 + rating / 5 * 0.35, 0, 1)), 4),
                "is_new_brand": is_new_brand,
                "is_new_item": is_new_item,
            }
        )
    return pd.DataFrame(rows)


def generate_users(seed: int, n_users: int = 80) -> pd.DataFrame:
    """Generate deterministic user preference profiles, including new users."""
    if n_users < 1:
        raise ValueError("n_users must be positive")

    rng = _rng(seed + 1)
    rows: list[dict[str, Any]] = []
    for index in range(n_users):
        category = CATEGORIES[index % len(CATEGORIES)]
        low, high = _CATEGORY_SPECS[category]["price_range"]
        budget_min = round(float(rng.uniform(low, (low + high) / 2)), 2)
        budget_max = round(float(rng.uniform((low + high) / 2, high)), 2)
        rows.append(
            {
                "user_id": f"u{index + 1:04d}",
                "preferred_category": category,
                "budget_min": budget_min,
                "budget_max": max(budget_max, budget_min + 1),
                "preferred_features": _feature_string(_CATEGORY_SPECS[category]["features"], rng),
                "is_new_user": index < max(2, n_users // 10),
            }
        )
    return pd.DataFrame(rows)


def generate_interactions(
    products: pd.DataFrame,
    users: pd.DataFrame,
    seed: int,
    n_interactions: int = 1200,
) -> pd.DataFrame:
    """Generate deterministic interactions while reserving new items without history."""
    if n_interactions < 0:
        raise ValueError("n_interactions must be non-negative")
    if products.empty or users.empty or n_interactions == 0:
        return pd.DataFrame(columns=["user_id", "product_id", "event_type", "interaction_strength"])

    rng = _rng(seed + 2)
    eligible_products = products.loc[~products["is_new_item"], "product_id"].to_numpy()
    rows = []
    event_types = np.array(list(_EVENT_STRENGTH))
    user_ids = users["user_id"].to_numpy()
    for _ in range(n_interactions):
        event_type = str(rng.choice(event_types, p=[0.62, 0.18, 0.14, 0.06]))
        rows.append(
            {
                "user_id": str(rng.choice(user_ids)),
                "product_id": str(rng.choice(eligible_products)),
                "event_type": event_type,
                "interaction_strength": _EVENT_STRENGTH[event_type],
            }
        )
    return pd.DataFrame(rows)


def generate_dataset(
    seed: int,
    output_dir: str | Path | None = None,
    n_products: int = 120,
    n_users: int = 80,
    n_interactions: int = 1200,
) -> dict[str, pd.DataFrame]:
    """Generate, validate, and optionally save all synthetic project tables."""
    products = generate_products(seed, n_products)
    users = generate_users(seed, n_users)
    interactions = generate_interactions(products, users, seed, n_interactions)
    dataset = {"products": products, "users": users, "interactions": interactions}

    errors = (
        validate_products(products)
        + validate_users(users)
        + validate_interactions(interactions)
    )
    if errors:
        raise ValueError("Generated dataset failed validation: " + "; ".join(errors))

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, table in dataset.items():
            table.to_csv(destination / f"{name}.csv", index=False)
    return dataset
