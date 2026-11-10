import numpy as np
import pandas as pd
import pytest

from src.data_generation import generate_products
from src.features import build_product_matrix, build_user_profile


def test_product_matrix_is_finite_and_matches_product_rows():
    products = generate_products(seed=42, n_products=12)

    matrix, encoder = build_product_matrix(products)

    assert matrix.shape[0] == len(products)
    assert matrix.shape[1] > 0
    assert np.isfinite(matrix).all()
    assert hasattr(encoder, "transform")


def test_new_user_profile_has_same_width_as_product_matrix():
    products = generate_products(seed=42, n_products=12)
    matrix, encoder = build_product_matrix(products)

    profile = build_user_profile(
        {
            "preferred_category": "Television",
            "budget_min": 400,
            "budget_max": 900,
            "preferred_features": "4K|HDR",
        },
        encoder,
    )

    assert profile.shape == (1, matrix.shape[1])
    assert np.isfinite(profile).all()


def test_malformed_budget_raises_a_useful_value_error():
    products = generate_products(seed=42, n_products=12)
    _, encoder = build_product_matrix(products)

    with pytest.raises(ValueError, match="budget_min and budget_max"):
        build_user_profile(
            {
                "preferred_category": "Phone",
                "budget_min": "not-a-number",
                "budget_max": 900,
                "preferred_features": "5G",
            },
            encoder,
        )


def test_missing_optional_product_metadata_uses_finite_defaults():
    products = generate_products(seed=42, n_products=6).drop(columns=["key_specifications"])

    matrix, _ = build_product_matrix(products)

    assert matrix.shape[0] == len(products)
    assert np.isfinite(matrix).all()
