from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allows `python3 src/eda.py` to work when executed directly.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, TABLES_DIR
from src.data_loader import load_weather_dataset
from src.preprocess import DATE_COL, add_date_features


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda() -> None:
    _ensure_dirs()
    df = load_weather_dataset()

    if DATE_COL in df.columns:
        df = add_date_features(df, DATE_COL)

    # 1) Basic statistics for key numeric columns
    numeric_cols = [
        "MinTemp",
        "MaxTemp",
        "Humidity9am",
        "Humidity3pm",
        "Rainfall",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Pressure9am",
        "Pressure3pm",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    basic_stats = df[numeric_cols].describe().T
    basic_stats.to_csv(TABLES_DIR / "eda_basic_stats_numeric.csv")

    # 2) Temperature distribution
    if "MaxTemp" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df["MaxTemp"].dropna(), bins=40, kde=True)
        plt.title("Distribution of MaxTemp")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eda_max_temp_distribution.png", dpi=200)
        plt.close()

    if "MinTemp" in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df["MinTemp"].dropna(), bins=40, kde=True)
        plt.title("Distribution of MinTemp")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eda_min_temp_distribution.png", dpi=200)
        plt.close()

    # 3) Rain frequency
    if "RainTomorrow" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="RainTomorrow", data=df)
        plt.title("RainTomorrow class distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eda_rain_tomorrow_distribution.png", dpi=200)
        plt.close()

    # 4) Monthly average temperature trend
    if "month" in df.columns and "MaxTemp" in df.columns:
        month_avg = df.groupby("month")["MaxTemp"].mean(numeric_only=True).reset_index()
        plt.figure(figsize=(9, 4))
        sns.lineplot(x="month", y="MaxTemp", data=month_avg, marker="o")
        plt.title("Average MaxTemp by Month")
        plt.xlabel("Month (1-12)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eda_avg_maxtemp_by_month.png", dpi=200)
        plt.close()

    # 5) Relationship: Humidity vs MaxTemp (colored by RainTomorrow)
    if all(c in df.columns for c in ["Humidity9am", "MaxTemp", "RainTomorrow"]):
        sample = df[[c for c in ["Humidity9am", "MaxTemp", "RainTomorrow"]]].dropna()
        if len(sample) > 50_000:
            sample = sample.sample(50_000, random_state=42)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x="Humidity9am",
            y="MaxTemp",
            hue="RainTomorrow",
            alpha=0.25,
            s=10,
        )
        plt.title("Humidity9am vs MaxTemp (colored by RainTomorrow)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eda_humidity_vs_maxtemp.png", dpi=200)
        plt.close()

    # 6) Correlations heatmap for numeric columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (selected numeric features)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=200)
        plt.close()

    print("EDA complete. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    run_eda()

