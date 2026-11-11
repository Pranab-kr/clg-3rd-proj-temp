"""Human-readable explanations for recommendation results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


def _number(product: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = product.get(name, default)
    try:
        return float(value) if pd.notna(value) else default
    except (TypeError, ValueError):
        return default


def build_recommendation_reason(
    product: pd.Series,
    preference_match: float,
    is_new_item: bool,
) -> str:
    """Build a reason from product metadata and the calculated match score.

    The text describes available metadata only; it does not infer purchases,
    reviews, or other unsupported customer behavior.
    """
    category = str(product.get("category", "product"))
    price = _number(product, "price")
    warranty = _number(product, "warranty_years")
    discount = _number(product, "discount")
    value_score = _number(product, "value_score")
    match = max(0.0, min(1.0, float(preference_match)))

    parts = [f"{category} with {match:.0%} preference similarity"]
    if price > 0:
        parts.append(f"priced at {price:.2f}")
    if warranty > 0:
        parts.append(f"{warranty:g}-year warranty")
    if discount > 0:
        parts.append(f"{discount:.0%} discount")
    if value_score >= 0.7:
        parts.append("strong value score")
    elif value_score > 0:
        parts.append("value score included in ranking")

    if is_new_item:
        parts.append("new item with no assumed interaction history")

    return "Recommended because it has " + ", ".join(parts) + "."


def build_popularity_reason(product: pd.Series) -> str:
    """Explain the documented signals used by the established-item baseline.

    Only signals present on the product row are mentioned; nothing is invented.
    """
    parts: list[str] = []
    activity = pd.to_numeric(product.get("activity"), errors="coerce")
    if pd.notna(activity):
        parts.append(f"{int(activity):,} recorded interactions")
    rating = pd.to_numeric(product.get("rating"), errors="coerce")
    if pd.notna(rating):
        parts.append(f"{rating:.1f} average rating")
    reviews = pd.to_numeric(product.get("review_count"), errors="coerce")
    if pd.notna(reviews):
        parts.append(f"{int(reviews):,} reviews")
    if not parts:
        return "Ranked by the baseline's historical interaction, rating, and review signals."
    return "Ranked by popularity signals: " + ", ".join(parts) + "."
