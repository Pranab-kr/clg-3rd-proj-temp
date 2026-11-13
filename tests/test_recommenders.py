import pandas as pd
import pytest

from src.recommenders import (
    cold_start_recommendations,
    popular_recommendations,
    recommend_for_user,
)


def _controlled_fixture():
    products = pd.DataFrame(
        [
            {
                "product_id": "established-tv",
                "product_name": "Established Television",
                "category": "Television",
                "brand": "OldBrand",
                "brand_age_years": 15,
                "product_age_years": 4,
                "price": 850.0,
                "discount": 0.05,
                "rating": 4.2,
                "review_count": 900,
                "warranty_years": 1,
                "key_specifications": "4K|Smart TV",
                "value_score": 0.55,
                "is_new_brand": False,
                "is_new_item": False,
            },
            {
                "product_id": "new-tv",
                "product_name": "New Brand Television",
                "category": "Television",
                "brand": "NewBrand",
                "brand_age_years": 1,
                "product_age_years": 0,
                "price": 650.0,
                "discount": 0.25,
                "rating": 4.8,
                "review_count": 0,
                "warranty_years": 3,
                "key_specifications": "4K|HDR",
                "value_score": 0.92,
                "is_new_brand": True,
                "is_new_item": True,
            },
            {
                "product_id": "phone",
                "product_name": "Other Category Phone",
                "category": "Phone",
                "brand": "PhoneBrand",
                "brand_age_years": 8,
                "product_age_years": 2,
                "price": 500.0,
                "discount": 0.10,
                "rating": 4.0,
                "review_count": 300,
                "warranty_years": 2,
                "key_specifications": "5G|OLED",
                "value_score": 0.60,
                "is_new_brand": False,
                "is_new_item": False,
            },
        ]
    )
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "product_id": "established-tv", "event_type": "view", "interaction_strength": 1.0},
            {"user_id": "u2", "product_id": "established-tv", "event_type": "purchase", "interaction_strength": 4.0},
            {"user_id": "u3", "product_id": "phone", "event_type": "view", "interaction_strength": 1.0},
        ]
    )
    users = pd.DataFrame(
        [
            {
                "user_id": "new-user",
                "preferred_category": "Television",
                "budget_min": 500.0,
                "budget_max": 800.0,
                "preferred_features": "4K|HDR",
                "is_new_user": True,
            }
        ]
    )
    return products, interactions, users


def test_popular_recommendations_return_documented_columns_and_rank_established_item():
    products, interactions, _ = _controlled_fixture()

    result = popular_recommendations(products, interactions, top_k=2)

    assert {"product_id", "product_name", "brand", "category", "score", "rank", "reason"} <= set(result.columns)
    assert len(result) == 2
    assert result.iloc[0]["product_id"] == "established-tv"
    assert result["rank"].tolist() == [1, 2]
    assert result["score"].is_monotonic_decreasing


def test_cold_start_recommendations_can_surface_suitable_zero_history_new_item():
    products, interactions, _ = _controlled_fixture()
    preferences = {
        "preferred_category": "Television",
        "budget_min": 500.0,
        "budget_max": 800.0,
        "preferred_features": "4K|HDR",
    }

    result = cold_start_recommendations(products, interactions, preferences, top_k=1)

    assert result.iloc[0]["product_id"] == "new-tv"
    assert bool(result.iloc[0]["is_new_item"])
    assert result.iloc[0]["exposure_component"] > 0
    assert result.iloc[0]["reason"]


def test_exposure_boost_requires_category_and_budget_suitability():
    products, interactions, _ = _controlled_fixture()
    products.loc[products["product_id"] == "phone", "is_new_item"] = True
    products.loc[products["product_id"] == "phone", "is_new_brand"] = True
    products.loc[products["product_id"] == "phone", "price"] = 2000.0
    preferences = {
        "preferred_category": "Television",
        "budget_min": 500.0,
        "budget_max": 800.0,
        "preferred_features": "4K|HDR",
    }

    result = cold_start_recommendations(
        products, interactions, preferences, top_k=3, exposure_weight=0.25
    ).set_index("product_id")

    assert result.loc["phone", "exposure_component"] == 0.0


def test_recommend_for_new_user_uses_preferences_without_interactions():
    products, _, users = _controlled_fixture()
    empty_interactions = pd.DataFrame(
        columns=["user_id", "product_id", "event_type", "interaction_strength"]
    )

    result = recommend_for_user("new-user", products, users, empty_interactions, top_k=2)

    assert set(result) == {"baseline", "proposed"}
    assert len(result["baseline"]) == 2
    assert len(result["proposed"]) == 2
    assert result["proposed"].iloc[0]["category"] == "Television"


def test_invalid_category_is_rejected_before_scoring():
    products, interactions, _ = _controlled_fixture()

    with pytest.raises(ValueError, match="preferred_category"):
        cold_start_recommendations(
            products,
            interactions,
            {
                "preferred_category": "Tablet",
                "budget_min": 500,
                "budget_max": 800,
                "preferred_features": "4K",
            },
        )


def test_empty_catalog_and_invalid_exposure_weight_are_rejected():
    _, interactions, _ = _controlled_fixture()
    preferences = {
        "preferred_category": "Television",
        "budget_min": 500,
        "budget_max": 800,
    }

    with pytest.raises(ValueError, match="at least one row"):
        popular_recommendations(pd.DataFrame(), interactions)
    with pytest.raises(ValueError, match="exposure_weight"):
        cold_start_recommendations(
            _controlled_fixture()[0], interactions, preferences, exposure_weight=-0.1
        )


def test_top_k_larger_than_catalog_returns_every_product_once_and_missing_review_is_safe():
    products, interactions, _ = _controlled_fixture()
    products = products.drop(columns=["review_count"])
    preferences = {
        "preferred_category": "Television",
        "budget_min": 500,
        "budget_max": 800,
    }

    popular = popular_recommendations(products, interactions, top_k=20)
    proposed = cold_start_recommendations(products, interactions, preferences, top_k=20)

    assert len(popular) == len(products)
    assert len(proposed) == len(products)
    assert proposed["product_id"].is_unique


def test_known_user_profile_can_be_used_when_interactions_exist():
    products, interactions, users = _controlled_fixture()

    result = recommend_for_user("missing-user", products, users, interactions)

    assert result["proposed"].empty is False


def test_missing_values_in_optional_numeric_metadata_use_safe_defaults():
    products, interactions, _ = _controlled_fixture()
    products.loc[0, "review_count"] = None
    products.loc[1, "review_count"] = None
    preferences = {
        "preferred_category": "Television",
        "budget_min": 500,
        "budget_max": 800,
    }

    baseline = popular_recommendations(products, interactions, top_k=3)
    proposed = cold_start_recommendations(products, interactions, preferences, top_k=3)

    assert len(baseline) == 3
    assert len(proposed) == 3
    assert baseline["score"].notna().all()
    assert proposed["score"].notna().all()
