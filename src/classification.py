"""Classical proxy-engagement classification for synthetic interactions.

The target is an interaction-derived engagement proxy, not purchase intent.
It labels wishlist, cart, and purchase events as engaged (1) and views as
non-engaged (0), using only user preferences and product metadata as inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


_USER_FEATURES = (
    "preferred_category",
    "budget_min",
    "budget_max",
    "preferred_features",
)
_PRODUCT_FEATURES = (
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
)
_REQUIRED_INTERACTION_COLUMNS = {"user_id", "product_id", "interaction_strength"}


def _require_columns(table: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def make_classification_dataset(
    users: pd.DataFrame, products: pd.DataFrame, interactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Join synthetic interactions to metadata and create proxy-engagement labels.

    The returned target is 1 for interaction strength at least 2 (wishlist,
    cart, or purchase) and 0 for a view. It must not be interpreted as a
    purchase-propensity label.
    """
    _require_columns(users, {"user_id", *_USER_FEATURES}, "users")
    _require_columns(products, {"product_id", *_PRODUCT_FEATURES}, "products")
    _require_columns(interactions, _REQUIRED_INTERACTION_COLUMNS, "interactions")

    interaction_frame = interactions.loc[:, ["user_id", "product_id", "interaction_strength"]].copy()
    strength = pd.to_numeric(interaction_frame["interaction_strength"], errors="coerce")
    if strength.isna().any():
        raise ValueError("interaction_strength must contain numeric values")

    merged = interaction_frame.merge(
        users.loc[:, ["user_id", *_USER_FEATURES]],
        on="user_id",
        how="left",
        validate="many_to_one",
    ).merge(
        products.loc[:, ["product_id", *_PRODUCT_FEATURES]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    feature_columns = [*_USER_FEATURES, *_PRODUCT_FEATURES]
    if merged[feature_columns].isna().any(axis=1).any():
        raise ValueError("interactions reference an unknown user or product")

    features = merged.loc[:, feature_columns].copy()
    labels = pd.Series((strength >= 2.0).astype(int).to_numpy(), name="engagement_proxy")
    return features, labels


def _classification_pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    categorical_columns = features.select_dtypes(
        include=["object", str, "category", "bool"]
    ).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]
    transformers = []
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if not transformers:
        raise ValueError("X must contain at least one feature column")
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", estimator)])


def _validate_training_inputs(X: pd.DataFrame, y: Sequence[int] | pd.Series) -> pd.Series:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.empty:
        raise ValueError("X must contain at least one row")
    labels = pd.Series(y).reset_index(drop=True)
    if len(X) != len(labels):
        raise ValueError("X and y must contain the same number of rows")
    if labels.isna().any() or labels.nunique() < 2:
        raise ValueError("y must contain both classes for classifier comparison")
    if len(X) < 8 or labels.value_counts().min() < 2:
        raise ValueError("classifier comparison needs at least eight rows and two examples of each class")
    return labels


def compare_classifiers(
    X: pd.DataFrame, y: pd.Series, seed: int = 42
) -> pd.DataFrame:
    """Compare four deterministic classical classifiers on a stratified holdout."""
    labels = _validate_training_inputs(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X.reset_index(drop=True),
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=min(5, len(X_train))),
    }
    rows = []
    for name, estimator in estimators.items():
        pipeline = _classification_pipeline(X, estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        rows.append(
            {
                "classifier": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def classification_report_table(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Return standard per-class metrics in a presentation-friendly table."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of rows")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict) or label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "label": str(label),
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": int(values["support"]),
            }
        )
    return pd.DataFrame(rows)
