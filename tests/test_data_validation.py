import pandas as pd

from src.data_validation import (
    validate_interactions,
    validate_products,
    validate_users,
)


def test_valid_generated_tables_have_no_validation_errors():
    products = pd.DataFrame(
        {
            "product_id": ["p1"],
            "product_name": ["Synthetic TV"],
            "category": ["Television"],
            "brand": ["Example"],
            "brand_age_years": [2],
            "product_age_years": [1],
            "price": [500.0],
            "discount": [0.1],
            "rating": [4.2],
            "review_count": [10],
            "warranty_years": [2],
            "key_specifications": ["4K"],
            "value_score": [0.8],
            "is_new_brand": [True],
            "is_new_item": [True],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": ["u1"],
            "preferred_category": ["Television"],
            "budget_min": [300.0],
            "budget_max": [700.0],
            "preferred_features": ["4K"],
            "is_new_user": [True],
        }
    )
    interactions = pd.DataFrame(
        {
            "user_id": ["u1"],
            "product_id": ["p1"],
            "event_type": ["view"],
            "interaction_strength": [1.0],
        }
    )

    assert validate_products(products) == []
    assert validate_users(users) == []
    assert validate_interactions(interactions) == []


def test_validation_reports_missing_columns_and_invalid_values():
    invalid_products = pd.DataFrame({"product_id": ["p1"], "price": [-10.0]})

    errors = validate_products(invalid_products)

    assert any("missing required columns" in error for error in errors)
    assert any("price" in error for error in errors)


def test_interaction_validation_reports_missing_columns_without_raising():
    errors = validate_interactions(pd.DataFrame())

    assert any("missing required columns" in error for error in errors)


def test_partial_interaction_validation_reports_schema_errors_without_raising():
    errors = validate_interactions(pd.DataFrame({"user_id": ["u1"]}))

    assert any("missing required columns" in error for error in errors)
