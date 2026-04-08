from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_col: str
    # If True, map string labels like "Yes"/"No" to 1/0
    classification: bool


TASK_TEMPERATURE = TaskSpec(name="temperature", target_col="MaxTemp", classification=False)
TASK_RAIN = TaskSpec(name="rain", target_col="RainTomorrow", classification=True)

DATE_COL = "Date"

# Known categorical columns for the weatherAUS dataset.
CATEGORICAL_CANDIDATES = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday",
]


def add_date_features(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """
    Adds simple date-derived numeric features and drops the original date column.

    weatherAUS uses day-first dates (e.g., '2015-01-01' or sometimes '01/01/2015'
    depending on preprocessing). We set `dayfirst=True` to be safe.
    """

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["day_of_week"] = out[date_col].dt.dayofweek

    out = out.drop(columns=[date_col])
    return out


def _map_yes_no_to_1_0(series: pd.Series) -> pd.Series:
    # Normalize common representations.
    s = series.astype("string").str.strip().str.lower()
    return s.map({"yes": 1, "no": 0})


def build_xy_for_task(df: pd.DataFrame, task: TaskSpec) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds (X, y) for either:
    - temperature regression: y = MaxTemp
    - rain classification: y = RainTomorrow (mapped to 0/1)
    """

    if DATE_COL in df.columns:
        df_fe = add_date_features(df, DATE_COL)
    else:
        # If the user preprocessed Date already, still proceed.
        df_fe = df.copy()

    if task.target_col not in df_fe.columns:
        raise KeyError(
            f"Expected target column `{task.target_col}` not found. "
            f"Available columns: {sorted(df_fe.columns)}"
        )

    # Drop rows where target is missing.
    df_fe = df_fe[df_fe[task.target_col].notna()].copy()

    y = df_fe[task.target_col]

    if task.classification:
        y = _map_yes_no_to_1_0(y)
        df_fe = df_fe[y.notna()].copy()
        y = y[y.notna()].astype(int)

    # Feature matrix drops the target column to avoid leakage.
    X = df_fe.drop(columns=[task.target_col])
    return X, y


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Returns (numeric_cols, categorical_cols).

    We treat known categorical candidates as categorical when present.
    Everything else is considered numeric (and will be coerced downstream).
    """

    categorical_cols = [c for c in CATEGORICAL_CANDIDATES if c in X.columns]
    # Also treat any explicit string/object columns as categorical.
    object_cols = [
        c
        for c in X.columns
        if is_object_dtype(X[c]) or is_categorical_dtype(X[c]) or is_string_dtype(X[c])
    ]
    categorical_cols = sorted(set(categorical_cols).union(object_cols))

    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols

