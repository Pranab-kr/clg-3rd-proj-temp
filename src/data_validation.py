"""Validation helpers for the synthetic project tables."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PRODUCT_REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "brand",
    "brand_age_years",
    "product_age_years",
    "price",
    "discount",
    "rating",
    "review_count",
    "warranty_years",
    "key_specifications",
    "value_score",
    "is_new_brand",
    "is_new_item",
}
USER_REQUIRED_COLUMNS = {
    "user_id",
    "preferred_category",
    "budget_min",
    "budget_max",
    "preferred_features",
    "is_new_user",
}
INTERACTION_REQUIRED_COLUMNS = {
    "user_id",
    "product_id",
    "event_type",
    "interaction_strength",
}


def _missing_columns(table: pd.DataFrame, required: Iterable[str]) -> list[str]:
    missing = sorted(set(required) - set(table.columns))
    return [f"missing required columns: {', '.join(missing)}"] if missing else []


def _numeric_range_error(table: pd.DataFrame, column: str, lower: float, upper: float | None = None) -> str | None:
    if column not in table:
        return None
    values = pd.to_numeric(table[column], errors="coerce")
    invalid = values.isna() | values.lt(lower)
    if upper is not None:
        invalid |= values.gt(upper)
    if invalid.any():
        limit = f" and <= {upper}" if upper is not None else ""
        return f"{column} must contain numeric values >= {lower}{limit}"
    return None


def _unique_id_error(table: pd.DataFrame, column: str) -> str | None:
    if column not in table:
        return None
    if table[column].isna().any() or table[column].duplicated().any():
        return f"{column} must contain non-null unique identifiers"
    return None


def validate_products(products: pd.DataFrame) -> list[str]:
    """Return readable schema and value errors for a products table."""
    errors = _missing_columns(products, PRODUCT_REQUIRED_COLUMNS)

    for column in ("product_id",):
        if (error := _unique_id_error(products, column)):
            errors.append(error)
    for column, lower, upper in (
        ("brand_age_years", 0, None),
        ("product_age_years", 0, None),
        ("price", 0.01, None),
        ("discount", 0, 0.9),
        ("rating", 1, 5),
        ("review_count", 0, None),
        ("warranty_years", 0, None),
        ("value_score", 0, 1),
    ):
        if error := _numeric_range_error(products, column, lower, upper):
            errors.append(error)
    if "category" in products and products["category"].isna().any():
        errors.append("category must not contain null values")
    return errors


def validate_users(users: pd.DataFrame) -> list[str]:
    """Return readable schema and value errors for a users table."""
    errors = _missing_columns(users, USER_REQUIRED_COLUMNS)

    if error := _unique_id_error(users, "user_id"):
        errors.append(error)
    for column in ("budget_min", "budget_max"):
        if error := _numeric_range_error(users, column, 0, None):
            errors.append(error)
    if {"budget_min", "budget_max"} <= set(users.columns):
        if (users["budget_min"] >= users["budget_max"]).any():
            errors.append("budget_min must be less than budget_max")
    return errors


def validate_interactions(interactions: pd.DataFrame) -> list[str]:
    """Return readable schema and value errors for an interactions table."""
    errors = _missing_columns(interactions, INTERACTION_REQUIRED_COLUMNS)

    for column in ("user_id", "product_id", "event_type"):
        if column in interactions and interactions[column].isna().any():
            errors.append(f"{column} must not contain null values")
    if error := _numeric_range_error(interactions, "interaction_strength", 0, None):
        errors.append(error)
    return errors
