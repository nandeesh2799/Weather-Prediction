from __future__ import annotations

from pathlib import Path
import random
import time

import pandas as pd
import requests

from src.config import WEATHER_CSV_PATH

CITY_COORDS = {
    "Bangalore": (12.9716, 77.5946),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
}
LEGACY_WEATHER_CSV_PATH = WEATHER_CSV_PATH.parent / "bangalore_weather.csv"


def _get_with_retry(
    url: str,
    params: dict,
    *,
    max_retries: int = 6,
    base_delay_seconds: float = 1.5,
    timeout_seconds: int = 60,
) -> requests.Response:
    """
    Executes GET with retries for transient failures and rate limits.
    Uses exponential backoff + jitter to reduce synchronized retries.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds)

            # Retry on explicit rate limit / temporary server issues.
            if response.status_code in {429, 500, 502, 503, 504}:
                response.raise_for_status()

            return response
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            sleep_seconds = base_delay_seconds * (2**attempt) + random.uniform(0, 0.8)

            # If server provides Retry-After, honor it.
            response_obj = getattr(exc, "response", None)
            if response_obj is not None:
                retry_after = response_obj.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_seconds = max(sleep_seconds, float(retry_after))
                    except ValueError:
                        pass

            print(
                f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                f"Retrying in {sleep_seconds:.1f}s..."
            )
            time.sleep(sleep_seconds)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected retry failure without captured exception.")


def _download_city_daily_weather(city: str, latitude: float, longitude: float) -> pd.DataFrame:
    """
    Downloads one city's daily weather history from Open-Meteo archive API
    and converts it into a simple ML-ready dataframe schema.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2000-01-01",
        "end_date": "2025-12-31",
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "windspeed_10m_max",
            ]
        ),
        "timezone": "Asia/Kolkata",
    }
    response = _get_with_retry(url, params, timeout_seconds=60)
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})

    city_df = pd.DataFrame(
        {
            "Date": daily.get("time", []),
            "MaxTemp": daily.get("temperature_2m_max", []),
            "MinTemp": daily.get("temperature_2m_min", []),
            "Rainfall": daily.get("precipitation_sum", []),
            "RainSum": daily.get("rain_sum", []),
            "WindGustSpeed": daily.get("windspeed_10m_max", []),
            "Location": city,
        }
    )

    # Derive simple same-day rain flag and next-day target.
    city_df["RainToday"] = (city_df["Rainfall"].fillna(0) > 0).map({True: "Yes", False: "No"})
    city_df["RainTomorrow"] = city_df["RainToday"].shift(-1)
    city_df = city_df.iloc[:-1].copy()  # Last row has unknown next-day label.
    return city_df


def _build_multicity_dataset_from_open_meteo() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for city, (latitude, longitude) in CITY_COORDS.items():
        try:
            frames.append(_download_city_daily_weather(city, latitude, longitude))
            print(f"Downloaded data for {city}.")
        except requests.exceptions.RequestException as exc:
            print(f"Skipping {city} due to download failure: {exc}")

    if not frames:
        raise RuntimeError("Failed to download weather data for all configured cities.")

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df = combined_df.sort_values(["Location", "Date"]).reset_index(drop=True)
    return combined_df


def load_weather_dataset(csv_path: Path = WEATHER_CSV_PATH) -> pd.DataFrame:
    """
    Loads the weather dataset into a dataframe.
    If `csv_path` does not exist, it tries legacy cached dataset first and then
    auto-downloads multi-city Indian weather from Open-Meteo.
    """
    if not csv_path.exists() and LEGACY_WEATHER_CSV_PATH.exists():
        print(
            f"Primary dataset missing at `{csv_path}`; "
            f"using legacy cached dataset `{LEGACY_WEATHER_CSV_PATH}`."
        )
        return pd.read_csv(LEGACY_WEATHER_CSV_PATH)

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_downloaded = _build_multicity_dataset_from_open_meteo()
        df_downloaded.to_csv(csv_path, index=False)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at `{csv_path}`.\n"
            "Could not auto-download multi-city dataset from Open-Meteo."
        )

    return pd.read_csv(csv_path)

