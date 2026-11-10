import pandas as pd

from src.explanations import build_recommendation_reason


def test_reason_mentions_actual_matching_signals_and_new_item_status():
    product = pd.Series(
        {
            "category": "Television",
            "price": 650.0,
            "discount": 0.25,
            "warranty_years": 3,
            "value_score": 0.92,
            "key_specifications": "4K|HDR",
            "is_new_item": True,
            "is_new_brand": True,
        }
    )

    reason = build_recommendation_reason(product, preference_match=0.9, is_new_item=True)

    assert "Television" in reason
    assert "budget" in reason.lower() or "price" in reason.lower()
    assert "warranty" in reason.lower()
    assert "value" in reason.lower()
    assert "new" in reason.lower()


def test_reason_does_not_claim_unsupported_purchases_or_reviews():
    product = pd.Series(
        {
            "category": "Phone",
            "price": 400.0,
            "discount": 0.0,
            "warranty_years": 1,
            "value_score": 0.5,
            "key_specifications": "5G",
            "is_new_item": False,
            "is_new_brand": False,
        }
    )

    reason = build_recommendation_reason(product, preference_match=0.2, is_new_item=False)

    assert "purchased" not in reason.lower()
    assert "reviewed" not in reason.lower()
    assert reason
