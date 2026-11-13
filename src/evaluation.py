"""Offline evaluation of the baseline and cold-start-aware recommenders.

All results computed here come from **synthetic** data and are illustrative only.

Evaluation protocol (simulated-time leave-one-out holdout):

* Interaction row order is treated as simulated time.
* For every user with at least two interactions and at least one positive
  interaction (``interaction_strength >= 2``: wishlist/cart/purchase), the
  user's *latest* positive interaction is held out as the relevant item. The
  rest of the interactions form the training history.
* Both methods are evaluated on the same users, the same training history, and
  the same K. Items a user already interacted with in training are removed
  from that user's list before the top-K cut.
* New items in the synthetic data have no interactions at all, so they can
  never appear as held-out positives. "Relevant new item" is therefore defined
  by content: a new item is relevant to a user when it is in the user's
  preferred category and its price lies within the user's budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TOP_K
from src.recommenders import cold_start_recommendations, popular_recommendations

POSITIVE_STRENGTH = 2.0


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    return int(k)


def precision_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Share of the K slots that hold a relevant item (short lists are not padded)."""
    k = _check_k(k)
    hits = sum(1 for item in recommended_ids[:k] if item in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Share of relevant items found in the top K; 0.0 when nothing is relevant."""
    k = _check_k(k)
    if not relevant_ids:
        return 0.0
    hits = len(set(recommended_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def catalog_coverage(recommended_lists: list[list[str]], catalog_size: int) -> float:
    """Fraction of the catalog that appears in at least one recommendation list."""
    if isinstance(catalog_size, bool) or catalog_size < 1:
        raise ValueError("catalog_size must be a positive integer")
    unique = {item for items in recommended_lists for item in items}
    return min(len(unique) / catalog_size, 1.0)


def new_item_exposure(recommended_lists: list[list[str]], new_item_ids: set[str]) -> float:
    """Share of all recommendation slots occupied by new items."""
    slots = sum(len(items) for items in recommended_lists)
    if slots == 0:
        return 0.0
    new_slots = sum(1 for items in recommended_lists for item in items if item in new_item_ids)
    return new_slots / slots


def split_holdout(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's latest positive on a product seen only once.

    A product must occur exactly once for that user so the held-out item is not
    still present in the training history. Users without such a positive are
    skipped rather than evaluated with an impossible target.
    """
    required = {"user_id", "product_id", "interaction_strength"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError("interactions missing required columns: " + ", ".join(missing))

    ordered = interactions.reset_index(drop=True)
    held_index = []
    for _, group in ordered.groupby("user_id", sort=True):
        if len(group) < 2:
            continue
        product_counts = group["product_id"].value_counts()
        eligible_positive_indices = []
        for index, row in group.iterrows():
            product_id = row["product_id"]
            if (
                row["interaction_strength"] >= POSITIVE_STRENGTH
                and product_counts[product_id] == 1
            ):
                eligible_positive_indices.append(index)
        if eligible_positive_indices:
            held_index.append(eligible_positive_indices[-1])
    held = ordered.loc[held_index].sort_values("user_id", kind="mergesort")
    train = ordered.drop(index=held_index)
    return train.reset_index(drop=True), held.reset_index(drop=True)


def _ranked_ids(frame: pd.DataFrame, seen: set[str], k: int) -> list[str]:
    ids = [pid for pid in frame["product_id"] if pid not in seen]
    return ids[:k]


def evaluate_recommenders(
    products: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Compare baseline and proposed recommenders on the same users and K."""
    top_k = _check_k(top_k)
    train, held = split_holdout(interactions)
    if held.empty:
        raise ValueError("no users have a held-out positive interaction to evaluate")

    catalog = len(products)
    ranked_baseline = popular_recommendations(products, train, top_k=catalog)
    new_ids = set(products.loc[products["is_new_item"].astype(bool), "product_id"])
    rating = products.set_index("product_id")["rating"]
    value = products.set_index("product_id")["value_score"]
    user_table = users.set_index("user_id")
    seen_by_user = train.groupby("user_id")["product_id"].agg(set).to_dict()

    lists: dict[str, list[list[str]]] = {"baseline": [], "proposed": []}
    precision = {"baseline": [], "proposed": []}
    recall = {"baseline": [], "proposed": []}
    relevant_new_hits = {"baseline": [], "proposed": []}

    for _, row in held.iterrows():
        user_id = row["user_id"]
        if user_id not in user_table.index:
            continue
        prefs = user_table.loc[user_id].to_dict()
        seen = seen_by_user.get(user_id, set())
        relevant = {row["product_id"]}
        proposed = cold_start_recommendations(products, train, prefs, top_k=catalog)
        rec = {
            "baseline": _ranked_ids(ranked_baseline, seen, top_k),
            "proposed": _ranked_ids(proposed, seen, top_k),
        }
        in_scope_new = set(
            products.loc[
                products["is_new_item"].astype(bool)
                & (products["category"] == prefs["preferred_category"])
                & (products["price"] >= prefs["budget_min"])
                & (products["price"] <= prefs["budget_max"]),
                "product_id",
            ]
        )
        for method, ids in rec.items():
            lists[method].append(ids)
            precision[method].append(precision_at_k(ids, relevant, top_k))
            recall[method].append(recall_at_k(ids, relevant, top_k))
            if in_scope_new:
                relevant_new_hits[method].append(len(in_scope_new & set(ids)) / len(in_scope_new))

    n_users = len(lists["baseline"])
    if n_users == 0:
        raise ValueError("no held-out users match the users table")

    rows = []
    for method in ("baseline", "proposed"):
        shown = [pid for ids in lists[method] for pid in ids]
        rows.append(
            {
                "method": method,
                "precision_at_k": float(np.mean(precision[method])),
                "recall_at_k": float(np.mean(recall[method])),
                "catalog_coverage": catalog_coverage(lists[method], catalog),
                "new_item_exposure": new_item_exposure(lists[method], new_ids),
                "relevant_new_item_rate": float(np.mean(relevant_new_hits[method])) if relevant_new_hits[method] else 0.0,
                "avg_rating": float(rating.reindex(shown).mean()) if shown else 0.0,
                "avg_value_score": float(value.reindex(shown).mean()) if shown else 0.0,
                "n_users": n_users,
                "k": top_k,
            }
        )
    return pd.DataFrame(rows)
