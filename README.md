# Weather Prediction and Forecast Dashboard

A complete machine learning mini-project for Indian city weather forecasting, with:

- **Regression** task: predict next weather pattern through `MaxTemp`
- **Classification** task: predict `RainTomorrow` (`Yes` / `No`)
- **Streamlit dashboard** for interactive what-if simulation and forecast insights

The project includes data loading, preprocessing, EDA, model training/evaluation, artifact saving, and UI inference in one workflow.

---

## Features

- Multi-city weather dataset pipeline (Open-Meteo archive API)
- Auto-retry download logic with exponential backoff for transient API failures
- Fallback to local legacy dataset cache when available
- Automated model comparison for both tasks
- Saved best models and feature metadata for deployment/inference
- EDA plots and model evaluation reports generated into `reports/`
- Streamlit app with:
  - prediction metrics
  - rain probability and confidence
  - what-if rainfall sensitivity simulation
  - prediction log download (CSV)
  - recommendations and risk score

---

## Project Structure

```text
Weather-Prediction/
├── app.py
├── requirements.txt
├── dataset/
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocess.py
│   ├── models.py
│   └── train_evaluate.py
├── models/
├── artifacts/
└── reports/
    ├── figures/
    └── tables/
```

---

## Tech Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Requests, Joblib

All Python dependencies are listed in `requirements.txt`.

---

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Behavior

The loader follows this order:

1. Use primary dataset path: `dataset/india_multicity_weather.csv` (if present)
2. Otherwise, use legacy cache: `dataset/bangalore_weather.csv` (if present)
3. Otherwise, download multi-city historical data from Open-Meteo and save to primary path

### Cities in default auto-download

- Bangalore
- Mumbai
- Delhi
- Chennai
- Hyderabad
- Kolkata
- Pune
- Ahmedabad

---

## Run Exploratory Data Analysis (EDA)

```bash
python3 src/eda.py
```

Outputs:

- Figures: `reports/figures/`
- Tables: `reports/tables/`

---

## Train and Evaluate Models

```bash
python3 src/train_evaluate.py
```

What this does:

- loads dataset
- prepares task-specific features/targets
- train/test split
- trains multiple candidate models
- evaluates:
  - Regression: `MAE`, `RMSE`, `R2`
  - Classification: `Accuracy`, `Precision`, `Recall`, `F1`
- saves best artifacts:
  - `models/best_temperature_max_temp.joblib`
  - `models/best_rain_rain_tomorrow.joblib`
  - `artifacts/feature_columns.json`
- writes report tables/plots under `reports/`

---

## Run the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## Troubleshooting

### `python3: can't open file 'train.py'`

Use:

```bash
python3 src/train_evaluate.py
```

### Open-Meteo rate limits (`429 Too Many Requests`) or temporary network errors

The project now includes retry + backoff logic in `src/data_loader.py`.
If issues persist:

- wait a minute and retry
- check internet/proxy settings
- keep a local dataset file in `dataset/` to avoid fresh API calls

### `ModuleNotFoundError` (for example `joblib`)

Make sure virtual environment is active and dependencies are installed:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproducibility Notes

- Train/test split is controlled through config in `src/config.py`
- Random seed is fixed (`RANDOM_STATE = 42`)
- Best models are selected by:
  - highest `R2` for temperature regression
  - highest `F1` for rain classification

---

## Future Improvements

- Add scheduled dataset refresh jobs
- Add hyperparameter tuning (Grid/Random/Bayesian search)
- Track experiments with MLflow
- Add Dockerfile and CI workflow for reproducible deployment
- Add unit tests for loader/preprocess/training modules

---

## License

Add a license file (`LICENSE`) before public release (MIT is a common choice for portfolio projects).

