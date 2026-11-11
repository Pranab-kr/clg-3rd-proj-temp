"""Smoke tests for the demo helpers behind app.py (no browser needed)."""

import pandas as pd
import pytest

from src.config import CATEGORIES, PROJECT_ROOT, TOP_K
from src.demo import (
    SYNTHETIC_NOTICE,
    feature_options,
    load_demo_data,
    load_saved_comparison,
    recommend_for_preferences,
    recommend_for_known_user,
)


@pytest.fixture(scope="module")
def data():
    return load_demo_data(seed=42, data_dir=None)


def test_load_demo_data_returns_validated_tables(data):
    assert set(data) == {"products", "users", "interactions"}
    assert len(data["products"]) == 120
    assert data["products"]["is_new_item"].any()
    assert "synthetic" in SYNTHETIC_NOTICE.lower()


def test_load_demo_data_prefers_saved_csv(tmp_path):
    first = load_demo_data(seed=7, data_dir=None)
    for name, table in first.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    loaded = load_demo_data(seed=42, data_dir=tmp_path)
    pd.testing.assert_frame_equal(
        loaded["products"].reset_index(drop=True),
        first["products"].reset_index(drop=True),
        check_dtype=False,
    )


def test_default_input_returns_both_tables_with_reasons(data):
    prefs = {
        "preferred_category": "Television",
        "budget_min": 300.0,
        "budget_max": 1500.0,
        "preferred_features": "4K|HDR",
    }
    result = recommend_for_preferences(data["products"], data["interactions"], prefs)
    for name in ("baseline", "proposed"):
        assert 0 < len(result[name]) <= TOP_K
        assert {"product_id", "score", "rank", "reason"} <= set(result[name].columns)
    assert result["proposed"]["reason"].str.strip().str.len().gt(0).all()
    assert (result["proposed"]["category"] == "Television").any()


def test_new_user_without_interactions_gets_recommendations(data):
    prefs = {
        "preferred_category": "Laptop",
        "budget_min": 500.0,
        "budget_max": 2000.0,
        "preferred_features": "SSD",
    }
    empty = data["interactions"].iloc[0:0]
    result = recommend_for_preferences(data["products"], empty, prefs)
    assert not result["proposed"].empty
    assert result["proposed"]["reason"].str.len().gt(0).all()


def test_invalid_budget_raises_clear_error(data):
    prefs = {"preferred_category": "Phone", "budget_min": 900.0, "budget_max": 100.0}
    with pytest.raises(ValueError, match="budget"):
        recommend_for_preferences(data["products"], data["interactions"], prefs)


def test_known_user_path(data):
    result = recommend_for_known_user("u0001", data)
    assert 0 < len(result["proposed"]) <= TOP_K
    assert 0 < len(result["baseline"]) <= TOP_K


def test_feature_options_per_category(data):
    for category in CATEGORIES:
        options = feature_options(data["products"], category)
        assert options and options == sorted(options)
        assert all("|" not in option for option in options)


def test_load_saved_comparison_missing_returns_none(tmp_path):
    assert load_saved_comparison(tmp_path) is None


def test_load_saved_comparison_reads_csv(tmp_path):
    pd.DataFrame({"method": ["baseline", "proposed"], "seed": [42, 42]}).to_csv(
        tmp_path / "recommender_comparison.csv", index=False
    )
    table = load_saved_comparison(tmp_path)
    assert list(table["method"]) == ["baseline", "proposed"]


def test_streamlit_app_renders_every_scenario_without_exception():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    for category in CATEGORIES:
        app.sidebar.selectbox[0].set_value(category).run()
        assert not app.exception
        assert not app.error
    app.sidebar.radio[0].set_value("Existing synthetic user").run()
    assert not app.exception


def test_new_user_baseline_uses_catalog_history_but_proposed_needs_none(data):
    from src.demo import recommend_for_new_user
    from src.recommenders import popular_recommendations

    prefs = {
        "preferred_category": "Television",
        "budget_min": 300.0,
        "budget_max": 1800.0,
        "preferred_features": "4K|HDR",
    }
    result = recommend_for_new_user(data, prefs)
    expected = popular_recommendations(data["products"], data["interactions"])
    pd.testing.assert_frame_equal(result["baseline"], expected)
    assert not result["baseline"]["is_new_item"].any()
    assert result["proposed"]["is_new_item"].any()
