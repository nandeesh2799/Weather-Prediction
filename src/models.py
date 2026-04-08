from __future__ import annotations

from typing import Dict, List

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Preprocessing for both regression and classification:
    - numeric: impute missing values (median) + scale
    - categorical: impute missing values (most frequent) + one-hot encode
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def build_regression_models(numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Pipeline]:
    """
    Returns regression model pipelines for predicting MaxTemp.
    """

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    return {
        "LinearRegression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "DecisionTreeRegressor": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DecisionTreeRegressor(random_state=42)),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
            ]
        ),
    }


def build_classification_models(
    numeric_cols: List[str], categorical_cols: List[str]
) -> Dict[str, Pipeline]:
    """
    Returns classification model pipelines for predicting RainTomorrow.
    """

    preprocessor = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "DecisionTreeClassifier": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }

