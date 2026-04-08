from __future__ import annotations

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allows `python3 src/train_evaluate.py` to work when executed directly.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BEST_RAIN_MODEL_PATH,
    BEST_TEMPERATURE_MODEL_PATH,
    FEATURES_METADATA_PATH,
    FIGURES_DIR,
    REPORTS_DIR,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.data_loader import load_weather_dataset
from src.models import build_classification_models, build_regression_models
from src.preprocess import TASK_RAIN, TASK_TEMPERATURE, build_xy_for_task, infer_feature_types


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def _plot_regression(y_true: np.ndarray, y_pred: np.ndarray, figure_path_prefix: Path) -> None:
    import matplotlib.pyplot as plt

    _ensure_parent_dir(figure_path_prefix)

    # True vs Pred scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.2, s=10)
    min_v = float(np.min([y_true.min(), y_pred.min()]))
    max_v = float(np.max([y_true.max(), y_pred.max()]))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2)
    plt.xlabel("True MaxTemp")
    plt.ylabel("Predicted MaxTemp")
    plt.title("Regression: True vs Predicted")
    plt.tight_layout()
    plt.savefig(figure_path_prefix.with_name(figure_path_prefix.name + "_true_vs_pred.png"), dpi=200)
    plt.close()

    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.2, s=10)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Predicted MaxTemp")
    plt.ylabel("Residual (True - Predicted)")
    plt.title("Regression: Residuals")
    plt.tight_layout()
    plt.savefig(figure_path_prefix.with_name(figure_path_prefix.name + "_residuals.png"), dpi=200)
    plt.close()


def _plot_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model,
    figure_path_prefix: Path,
) -> None:
    import matplotlib.pyplot as plt

    _ensure_parent_dir(figure_path_prefix)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(cmap="Blues", values_format="d")
    plt.title("Classification: Confusion Matrix (RainTomorrow)")
    plt.tight_layout()
    plt.savefig(figure_path_prefix.with_name(figure_path_prefix.name + "_confusion_matrix.png"), dpi=200)
    plt.close()


def _plot_roc_curve(y_true: np.ndarray, y_prob_yes: np.ndarray, figure_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay

    plt.figure(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob_yes, name="ROC").plot()
    plt.title("Classification: ROC Curve (RainTomorrow)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()


def train_and_evaluate() -> None:
    # Load data
    df = load_weather_dataset()

    # Build datasets for each task
    X_temp, y_temp = build_xy_for_task(df, TASK_TEMPERATURE)
    X_rain, y_rain = build_xy_for_task(df, TASK_RAIN)

    # Train/test split (use consistent random seed)
    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_rain_train, X_rain_test, y_rain_train, y_rain_test = train_test_split(
        X_rain, y_rain, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_rain
    )

    # Infer feature types from training data
    temp_numeric_cols, temp_categorical_cols = infer_feature_types(X_temp_train)
    rain_numeric_cols, rain_categorical_cols = infer_feature_types(X_rain_train)

    # Save metadata used by the UI
    _ensure_parent_dir(FEATURES_METADATA_PATH)
    metadata = {
        "temperature": {
            "feature_columns": list(X_temp.columns),
            "numeric_cols": temp_numeric_cols,
            "categorical_cols": temp_categorical_cols,
            "target_col": TASK_TEMPERATURE.target_col,
        },
        "rain": {
            "feature_columns": list(X_rain.columns),
            "numeric_cols": rain_numeric_cols,
            "categorical_cols": rain_categorical_cols,
            "target_col": TASK_RAIN.target_col,
        },
    }
    FEATURES_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Build model sets
    regression_models = build_regression_models(
        numeric_cols=temp_numeric_cols, categorical_cols=temp_categorical_cols
    )
    classification_models = build_classification_models(
        numeric_cols=rain_numeric_cols, categorical_cols=rain_categorical_cols
    )

    # Train + evaluate regression models
    regression_rows = []
    best_temp_model_name = None
    best_temp_metric = -np.inf
    best_temp_model = None
    best_temp_pred = None

    for name, model in regression_models.items():
        model.fit(X_temp_train, y_temp_train)
        preds = model.predict(X_temp_test)
        metrics = regression_metrics(y_temp_test.to_numpy(), preds)
        regression_rows.append({"Model": name, **metrics})
        if metrics["R2"] > best_temp_metric:
            best_temp_metric = metrics["R2"]
            best_temp_model_name = name
            best_temp_model = model
            best_temp_pred = preds

    regression_df = pd.DataFrame(regression_rows).sort_values("R2", ascending=False)
    regression_df.to_csv(REPORTS_DIR / "tables" / "temperature_model_metrics.csv", index=False)

    # Save best regression model
    _ensure_parent_dir(BEST_TEMPERATURE_MODEL_PATH)
    joblib.dump(best_temp_model, BEST_TEMPERATURE_MODEL_PATH)

    _plot_regression(
        y_true=y_temp_test.to_numpy(),
        y_pred=best_temp_pred,
        figure_path_prefix=FIGURES_DIR / "temperature_best_model",
    )

    # Train + evaluate classification models
    classification_rows = []
    best_rain_model_name = None
    best_rain_metric = -np.inf
    best_rain_model = None
    best_rain_pred = None
    best_rain_prob_yes = None

    for name, model in classification_models.items():
        model.fit(X_rain_train, y_rain_train)
        preds = model.predict(X_rain_test)
        metrics = classification_metrics(y_rain_test.to_numpy(), preds)
        classification_rows.append({"Model": name, **metrics})
        if metrics["F1"] > best_rain_metric:
            best_rain_metric = metrics["F1"]
            best_rain_model_name = name
            best_rain_model = model
            best_rain_pred = preds
            if hasattr(model, "predict_proba"):
                best_rain_prob_yes = model.predict_proba(X_rain_test)[:, 1]

    classification_df = pd.DataFrame(classification_rows).sort_values("F1", ascending=False)
    classification_df.to_csv(REPORTS_DIR / "tables" / "rain_model_metrics.csv", index=False)

    # Save best classification model
    _ensure_parent_dir(BEST_RAIN_MODEL_PATH)
    joblib.dump(best_rain_model, BEST_RAIN_MODEL_PATH)

    # Plots for best classification model
    _plot_classification(
        y_true=y_rain_test.to_numpy(),
        y_pred=best_rain_pred,
        model=best_rain_model,
        figure_path_prefix=FIGURES_DIR / "rain_best_model",
    )
    if best_rain_prob_yes is not None:
        _plot_roc_curve(
            y_true=y_rain_test.to_numpy(),
            y_prob_yes=best_rain_prob_yes,
            figure_path=FIGURES_DIR / "rain_best_model_roc_curve.png",
        )

    # Combined summary to make submission easy
    summary_path = REPORTS_DIR / "tables" / "combined_metrics_summary.csv"
    combined = {
        "best_temperature_model": [best_temp_model_name],
        "best_temperature_r2": [best_temp_metric],
        "best_rain_model": [best_rain_model_name],
        "best_rain_f1": [best_rain_metric],
    }
    pd.DataFrame(combined).to_csv(summary_path, index=False)

    print("Training complete.")
    print("Best temperature model:", best_temp_model_name, "R2:", best_temp_metric)
    print("Best rain model:", best_rain_model_name, "F1:", best_rain_metric)
    print("Saved:")
    print("-", BEST_TEMPERATURE_MODEL_PATH)
    print("-", BEST_RAIN_MODEL_PATH)
    print("-", FEATURES_METADATA_PATH)


if __name__ == "__main__":
    train_and_evaluate()

