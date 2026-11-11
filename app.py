"""Streamlit demo: baseline vs explainable cold-start-aware recommendations.

Thin entry point. All data loading and recommendation logic lives in src/.
"""

import pandas as pd
import streamlit as st

from src.config import CATEGORIES, RANDOM_SEED, TOP_K
from src.demo import (
    SYNTHETIC_NOTICE,
    build_preferences,
    feature_options,
    load_demo_data,
    load_saved_comparison,
    price_bounds,
    recommend_for_known_user,
    recommend_for_new_user,
)

st.set_page_config(page_title="Cold-Start Electronics Recommender", layout="wide")


@st.cache_data(show_spinner=False)
def _data() -> dict[str, pd.DataFrame]:
    return load_demo_data(RANDOM_SEED)


def _label_new(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a readable tag column for new-brand / zero-history products."""
    out = frame.copy()
    if "is_new_item" in out:
        out["tag"] = out["is_new_item"].map({True: "New item (no history)", False: ""})
    return out


def _show_tables(result: dict[str, pd.DataFrame]) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Baseline: popularity / history")
        st.caption("Ranks by interaction activity, rating, and review count.")
        base = result["baseline"]
        st.dataframe(
            base[["rank", "product_name", "brand", "category", "score"]],
            hide_index=True,
        )
        for _, row in base.iterrows():
            st.markdown(f"**{row['rank']}. {row['product_name']}** — {row['reason']}")
    with right:
        st.subheader("Proposed: content / value, cold-start aware")
        st.caption("Matches preferences and value; small bounded boost for new items.")
        prop = _label_new(result["proposed"])
        cols = ["rank", "product_name", "brand", "category", "score"]
        if "tag" in prop:
            cols.append("tag")
        st.dataframe(prop[cols], hide_index=True)
        for _, row in prop.iterrows():
            badge = " 🆕" if row.get("is_new_item", False) else ""
            st.markdown(f"**{row['rank']}. {row['product_name']}**{badge} — {row['reason']}")

    new_in_baseline = int(result["baseline"].get("is_new_item", pd.Series(dtype=bool)).sum())
    new_in_proposed = int(result["proposed"].get("is_new_item", pd.Series(dtype=bool)).sum())
    st.info(
        f"New zero-history items in the top {len(result['proposed'])}: "
        f"baseline {new_in_baseline}, proposed {new_in_proposed}."
    )


def _comparison_panel() -> None:
    st.header("Saved experiment comparison")
    table = load_saved_comparison()
    if table is None:
        st.warning(
            "No saved results found. Run `python scripts/run_experiments.py --seed 42` "
            "to generate them."
        )
        return
    st.caption(
        "Produced by scripts/run_experiments.py on synthetic data (leave-one-out, "
        f"K={int(table['k'].iloc[0]) if 'k' in table else TOP_K})."
    )
    st.dataframe(table.round(4), hide_index=True)
    st.markdown(
        "**Read honestly:** the proposed method is *not* more accurate on Precision/Recall "
        "here, because the synthetic history is random and popularity predicts a random "
        "held-out item best. Its advantage is on cold-start goals: coverage, new-item "
        "exposure, and surfacing relevant new items."
    )


def main() -> None:
    data = _data()
    products = data["products"]

    st.title("Cold-Start-Aware Electronics Recommender")
    st.warning(SYNTHETIC_NOTICE)
    st.markdown(
        "A history-based recommender can't recommend a **new product that nobody has "
        "interacted with yet**, and knows nothing about a **new user**. This demo "
        "compares it with an explainable content/value-based recommender that works "
        "from product metadata and stated preferences."
    )

    mode = st.sidebar.radio("Scenario", ["New user (preferences)", "Existing synthetic user"])
    exposure = st.sidebar.slider(
        "New-item exposure weight", 0.0, 0.25, 0.10, 0.01,
        help="Bounded boost for suitable zero-history new items (max 0.25).",
    )

    if mode == "New user (preferences)":
        category = st.sidebar.selectbox("Category", CATEGORIES)
        low, high = price_bounds(products, category)
        budget = st.sidebar.slider(
            "Budget", int(low), int(high) + 1, (int(low), int(high) + 1), step=10
        )
        features = st.sidebar.multiselect("Preferred features", feature_options(products, category))
        st.caption(
            "This user has no history. The baseline shows catalog-wide popularity; "
            "the proposed method uses only the preferences below."
        )
        if budget[0] >= budget[1]:
            st.error("Budget minimum must be below the maximum.")
        else:
            prefs = build_preferences(category, budget[0], budget[1], features)
            try:
                result = recommend_for_new_user(data, prefs, exposure_weight=exposure)
                _show_tables(result)
            except ValueError as exc:
                st.error(str(exc))
    else:
        users = data["users"]
        user_id = st.sidebar.selectbox("Synthetic user", users["user_id"].tolist())
        row = users.loc[users["user_id"] == user_id].iloc[0]
        st.caption(
            f"{user_id}: prefers {row['preferred_category']}, budget "
            f"{row['budget_min']:.0f}–{row['budget_max']:.0f}, features "
            f"{row['preferred_features']}"
            + (" (marked as a new user)" if row.get("is_new_user", False) else "")
        )
        _show_tables(recommend_for_known_user(user_id, data))

    st.divider()
    _comparison_panel()


main()
