import pandas as pd

from src.data_generation import (
    generate_dataset,
    generate_interactions,
    generate_products,
    generate_users,
)


PRODUCT_COLUMNS = {
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
USER_COLUMNS = {
    "user_id",
    "preferred_category",
    "budget_min",
    "budget_max",
    "preferred_features",
    "is_new_user",
}
INTERACTION_COLUMNS = {
    "user_id",
    "product_id",
    "event_type",
    "interaction_strength",
}


def test_generated_tables_have_required_schema_and_valid_ranges():
    products = generate_products(seed=42, n_products=30)
    users = generate_users(seed=42, n_users=20)
    interactions = generate_interactions(products, users, seed=42, n_interactions=150)

    assert PRODUCT_COLUMNS <= set(products.columns)
    assert USER_COLUMNS <= set(users.columns)
    assert INTERACTION_COLUMNS <= set(interactions.columns)
    assert products["product_id"].is_unique
    assert users["user_id"].is_unique
    assert products["price"].gt(0).all()
    assert products["discount"].between(0, 0.9).all()
    assert products["rating"].between(1, 5).all()
    assert users["budget_min"].lt(users["budget_max"]).all()
    assert interactions["interaction_strength"].ge(0).all()


def test_dataset_contains_controlled_new_item_and_new_user_cases():
    dataset = generate_dataset(seed=42, output_dir=None, n_products=60, n_users=30, n_interactions=300)
    products = dataset["products"]
    users = dataset["users"]
    interactions = dataset["interactions"]

    interacted_product_ids = set(interactions["product_id"])
    new_items_without_history = products.loc[
        products["is_new_item"] & ~products["product_id"].isin(interacted_product_ids)
    ]

    assert not new_items_without_history.empty
    assert users["is_new_user"].any()


def test_generation_is_reproducible_for_same_seed():
    first = generate_dataset(seed=7, output_dir=None, n_products=24, n_users=12, n_interactions=80)
    second = generate_dataset(seed=7, output_dir=None, n_products=24, n_users=12, n_interactions=80)

    for table_name in ("products", "users", "interactions"):
        pd.testing.assert_frame_equal(first[table_name], second[table_name])


def test_small_catalog_keeps_at_least_one_product_available_for_history():
    products = generate_products(seed=3, n_products=3)
    users = generate_users(seed=3, n_users=2)

    interactions = generate_interactions(products, users, seed=3, n_interactions=4)

    assert not interactions.empty
    assert (~products["is_new_item"]).any()
