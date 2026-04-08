from __future__ import annotations

import json
from datetime import date as date_type
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    BEST_RAIN_MODEL_PATH,
    BEST_TEMPERATURE_MODEL_PATH,
    FEATURES_METADATA_PATH,
)
from src.preprocess import DATE_COL, add_date_features

CITY_OPTIONS = [
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Other (type manually)",
]


def _load_required(path, kind: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing `{kind}` at `{path}`.\n"
            f"Run `python3 src/train_evaluate.py` first."
        )
    return path


@st.cache_resource(show_spinner=False)
def load_artifacts():
    _load_required(BEST_TEMPERATURE_MODEL_PATH, "best temperature model")
    _load_required(BEST_RAIN_MODEL_PATH, "best rain model")
    _load_required(FEATURES_METADATA_PATH, "feature metadata")

    temp_model = joblib.load(BEST_TEMPERATURE_MODEL_PATH)
    rain_model = joblib.load(BEST_RAIN_MODEL_PATH)

    metadata = json.loads(FEATURES_METADATA_PATH.read_text(encoding="utf-8"))
    return temp_model, rain_model, metadata


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def build_model_input_row(
    raw_inputs: Dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Converts user inputs into the feature dataframe expected by a trained pipeline.

    - Adds derived date features via `add_date_features`
    - Ensures all required `feature_columns` exist (missing ones become NaN)
    """

    # Start from a raw dataframe containing `Date` + user-provided columns.
    df_raw = pd.DataFrame([raw_inputs])

    # Add year/month/day/day_of_week and drop `Date`.
    if DATE_COL in df_raw.columns:
        df_fe = add_date_features(df_raw, DATE_COL)
    else:
        df_fe = df_raw.copy()

    # Ensure all required columns exist.
    for col in feature_columns:
        if col not in df_fe.columns:
            df_fe[col] = np.nan

    # Select and return exactly the expected columns.
    return df_fe[feature_columns]


def _get_rain_probability(rain_model, X_rain: pd.DataFrame) -> Optional[float]:
    if not hasattr(rain_model, "predict_proba"):
        return None
    try:
        return float(rain_model.predict_proba(X_rain)[0, 1])
    except Exception:
        return None


def _confidence_level(prob_yes: Optional[float]) -> tuple[str, float]:
    if prob_yes is None:
        return "Unknown", 0.0
    confidence = float(max(prob_yes, 1.0 - prob_yes))
    if confidence >= 0.8:
        return "High", confidence
    if confidence >= 0.65:
        return "Medium", confidence
    return "Low", confidence


def _build_what_if_table(
    raw_inputs: Dict[str, Any],
    temp_model,
    rain_model,
    temp_feature_columns: list[str],
    rain_feature_columns: list[str],
) -> pd.DataFrame:
    rows = []
    base_rainfall = float(raw_inputs.get("Rainfall", 0.0))
    # Simulate +/- around current rainfall.
    scenarios = np.linspace(max(0.0, base_rainfall - 20.0), base_rainfall + 20.0, 9)
    for rf in scenarios:
        candidate = dict(raw_inputs)
        candidate["Rainfall"] = float(rf)

        X_temp = build_model_input_row(candidate, temp_feature_columns)
        X_rain = build_model_input_row(candidate, rain_feature_columns)
        pred_temp = float(temp_model.predict(X_temp)[0])
        pred_rain = int(rain_model.predict(X_rain)[0])
        prob_yes = _get_rain_probability(rain_model, X_rain)

        rows.append(
            {
                "Rainfall_mm": round(float(rf), 2),
                "Pred_MaxTemp_C": round(pred_temp, 2),
                "Pred_RainTomorrow": "Yes" if pred_rain == 1 else "No",
                "Prob_RainTomorrow_Yes": None if prob_yes is None else round(prob_yes, 3),
            }
        )
    return pd.DataFrame(rows)


def _apply_theme(theme_name: str) -> None:
    if theme_name == "Aurora":
        bg = "linear-gradient(120deg, #0f172a 0%, #1d4ed8 40%, #0ea5e9 100%)"
        card = "rgba(255,255,255,0.08)"
        txt = "#e2e8f0"
    elif theme_name == "Sunset":
        bg = "linear-gradient(120deg, #3f0d12 0%, #a71d31 45%, #ff8c42 100%)"
        card = "rgba(255,255,255,0.10)"
        txt = "#fff7ed"
    else:
        bg = "linear-gradient(120deg, #0b132b 0%, #1c2541 40%, #3a506b 100%)"
        card = "rgba(255,255,255,0.08)"
        txt = "#e5e7eb"

    st.markdown(
        f"""
        <style>
        /* Hide Streamlit default chrome elements */
        [data-testid="stAppDeployButton"] {{
            display: none;
        }}
        #MainMenu {{
            visibility: hidden;
        }}
        footer {{
            visibility: hidden;
        }}
        .stApp {{
            background: {bg};
            color: {txt};
        }}
        div[data-testid="stMetric"] {{
            background: {card};
            border-radius: 14px;
            padding: 8px 12px;
            border: 1px solid rgba(255,255,255,0.15);
        }}
        div[data-testid="stDataFrame"] {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _generate_recommendations(pred_temp: float, prob_yes: Optional[float], wind_gust_speed: float) -> list[str]:
    recs: list[str] = []
    if pred_temp >= 36:
        recs.append("High heat expected: hydrate and avoid long direct-sun exposure.")
    elif pred_temp <= 15:
        recs.append("Cool conditions expected: carry a light jacket.")
    if prob_yes is not None and prob_yes >= 0.6:
        recs.append("High rain chance: keep an umbrella or raincoat ready.")
    if wind_gust_speed >= 45:
        recs.append("Strong winds likely: avoid unsecured outdoor items.")
    if not recs:
        recs.append("Weather looks fairly stable; standard outdoor planning should be fine.")
    return recs


def main():
    st.set_page_config(page_title="Weather Intelligence AI", page_icon="🌦️", layout="wide")

    try:
        temp_model, rain_model, metadata = load_artifacts()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    temp_meta = metadata["temperature"]
    rain_meta = metadata["rain"]

    temp_feature_columns = temp_meta["feature_columns"]
    rain_feature_columns = rain_meta["feature_columns"]

    with st.sidebar:
        st.header("Forecast Controls")
        st.caption("Choose location, style, and weather inputs.")
        theme = st.selectbox("Theme", ["Midnight", "Aurora", "Sunset"], index=1)
        city_pick = st.selectbox("Select City", CITY_OPTIONS, index=0)
        if city_pick == "Other (type manually)":
            location = st.text_input("Custom Location", value="Bangalore")
        else:
            location = city_pick

        # Date input
        dt = st.date_input("Date", value=date_type.today())
        dt_str = dt.strftime("%Y-%m-%d")

        rain_today = st.selectbox("RainToday", options=["No", "Yes"], index=0)

        # Temperatures
        min_temp = st.slider("MinTemp (today, C)", min_value=5.0, max_value=35.0, value=20.0, step=0.5)
        max_temp = st.slider("MaxTemp (today, C)", min_value=10.0, max_value=45.0, value=30.0, step=0.5)

        rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
        wind_gust_speed = st.slider("WindGustSpeed (km/h)", min_value=0.0, max_value=120.0, value=25.0, step=1.0)
        run_button = st.button("Run Forecast", type="primary", use_container_width=True)

    _apply_theme(theme)
    st.title("Weather Intelligence Dashboard")
    st.caption("Advanced ML UI with multi-location forecasting, confidence, simulation, and recommendations.")

    # Build raw inputs dict (what `add_date_features` expects).
    raw_inputs: Dict[str, Any] = {
        DATE_COL: dt_str,
        "Location": location,
        "MinTemp": float(min_temp),
        "MaxTemp": float(max_temp),
        "Rainfall": float(rainfall),
        "WindGustSpeed": float(wind_gust_speed),
        "RainToday": rain_today,
    }

    if not run_button:
        st.info("Set your forecast inputs in the sidebar and click `Run Forecast`.")
        return

    X_temp = build_model_input_row(raw_inputs, temp_feature_columns)
    X_rain = build_model_input_row(raw_inputs, rain_feature_columns)
    pred_temp = float(temp_model.predict(X_temp)[0])
    pred_rain = int(rain_model.predict(X_rain)[0])
    prob_yes = _get_rain_probability(rain_model, X_rain)
    rain_label = "Yes" if pred_rain == 1 else "No"
    conf_label, conf_value = _confidence_level(prob_yes)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Predicted MaxTemp", f"{pred_temp:.2f} C")
    with top2:
        st.metric("Predicted RainTomorrow", rain_label)
    with top3:
        st.metric("Rain Probability", "N/A" if prob_yes is None else f"{prob_yes*100:.1f}%")
    with top4:
        st.metric("Model Confidence", conf_label)

    st.progress(conf_value)
    st.caption("Confidence indicates how far prediction probability is from 50%.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Forecast Details", "What-If Simulator", "Prediction Log", "Recommendations"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Input Snapshot")
            st.dataframe(pd.DataFrame([raw_inputs]), use_container_width=True)
        with c2:
            st.subheader("Model Metadata")
            st.write(
                {
                    "temperature_model_features": len(temp_feature_columns),
                    "rain_model_features": len(rain_feature_columns),
                    "temperature_target": temp_meta.get("target_col"),
                    "rain_target": rain_meta.get("target_col"),
                }
            )

    with tab2:
        st.subheader("Rainfall Sensitivity Analysis")
        what_if_df = _build_what_if_table(
            raw_inputs=raw_inputs,
            temp_model=temp_model,
            rain_model=rain_model,
            temp_feature_columns=temp_feature_columns,
            rain_feature_columns=rain_feature_columns,
        )
        st.line_chart(what_if_df.set_index("Rainfall_mm")["Pred_MaxTemp_C"])
        if what_if_df["Prob_RainTomorrow_Yes"].notna().any():
            st.line_chart(what_if_df.set_index("Rainfall_mm")["Prob_RainTomorrow_Yes"])
        st.dataframe(what_if_df, use_container_width=True)

    with tab3:
        log_row = {
            "Date": dt_str,
            "Location": location,
            "MinTemp": float(min_temp),
            "MaxTemp_input": float(max_temp),
            "Rainfall": float(rainfall),
            "WindGustSpeed": float(wind_gust_speed),
            "RainToday": rain_today,
            "Pred_MaxTemp": round(pred_temp, 2),
            "Pred_RainTomorrow": rain_label,
            "Prob_RainTomorrow_Yes": None if prob_yes is None else round(prob_yes, 3),
        }
        if "prediction_log" not in st.session_state:
            st.session_state["prediction_log"] = []
        st.session_state["prediction_log"].append(log_row)

        log_df = pd.DataFrame(st.session_state["prediction_log"])
        st.dataframe(log_df.tail(20), use_container_width=True)
        st.download_button(
            "Download Prediction Log (CSV)",
            data=log_df.to_csv(index=False).encode("utf-8"),
            file_name="weather_prediction_log.csv",
            mime="text/csv",
        )

    with tab4:
        st.subheader("Smart Weather Recommendations")
        recommendations = _generate_recommendations(pred_temp, prob_yes, float(wind_gust_speed))
        for idx, item in enumerate(recommendations, start=1):
            st.write(f"{idx}. {item}")
        risk_score = 0.35 * min(pred_temp / 45.0, 1.0)
        risk_score += 0.45 * (0.0 if prob_yes is None else prob_yes)
        risk_score += 0.20 * min(float(wind_gust_speed) / 100.0, 1.0)
        st.metric("Overall Weather Risk Score", f"{risk_score*100:.1f}/100")


if __name__ == "__main__":
    main()

