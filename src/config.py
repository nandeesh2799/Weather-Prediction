from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Expected dataset file name (multi-city India historical weather).
WEATHER_CSV_PATH = DATA_DIR / "india_multicity_weather.csv"

# Training outputs
BEST_TEMPERATURE_MODEL_PATH = MODELS_DIR / "best_temperature_max_temp.joblib"
BEST_RAIN_MODEL_PATH = MODELS_DIR / "best_rain_rain_tomorrow.joblib"

# Store metadata needed by the app.
FEATURES_METADATA_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Basic split settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

