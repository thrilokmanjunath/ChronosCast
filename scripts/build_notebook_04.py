"""Generate 04_statistical_forecasting.ipynb."""

from pathlib import Path

import nbformat as nbf

notebooks_dir = Path("notebooks")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "language_info": {"name": "python", "version": "3.12"},
    "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
}

cells = [
    nbf.v4.new_markdown_cell("""# ChronosCast — Notebook 04: Statistical Forecasting Baselines
**Project**: ChronosCast: Intelligent Time Series Trend Analysis & Forecasting Platform

### Models Benchmarked:
1. **Level 1 Baselines**:
   - Naive (Persistence)
   - Seasonal Naive ($s=7$)
   - Moving Average ($w=7$)
2. **Level 2 Statistical Methods**:
   - Simple Exponential Smoothing (SES)
   - Holt's Linear Trend
   - Holt-Winters Seasonal Exponential Smoothing (Additive)
   - ARIMA & SARIMA
"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from chronoscast.models.baselines import NaiveForecaster, SeasonalNaiveForecaster, MovingAverageForecaster
from chronoscast.models.statistical import ExponentialSmoothingForecaster, ARIMAForecaster

df = pd.read_parquet("../data/interim/cleaned_sales.parquet")
# Fit on 2013-2016 for Store 1, Item 1 as representative benchmark series
s1i1_train = df[(df["store"] == 1) & (df["item"] == 1) & (df["date"] <= "2016-12-31")].copy()
s1i1_val = df[(df["store"] == 1) & (df["item"] == 1) & (df["date"] >= "2017-01-01") & (df["date"] <= "2017-01-28")].copy()
print(f"Train rows: {len(s1i1_train)}, Val rows (28 days): {len(s1i1_val)}")
"""),
    nbf.v4.new_markdown_cell("### 1. Fit & Forecast Baselines"),
    nbf.v4.new_code_cell("""naive = NaiveForecaster().fit(s1i1_train)
snaive = SeasonalNaiveForecaster(season_length=7).fit(s1i1_train)
ma7 = MovingAverageForecaster(window=7).fit(s1i1_train)

pred_naive = naive.predict(horizon=28)
pred_snaive = snaive.predict(horizon=28)
pred_ma7 = ma7.predict(horizon=28)
print("Baselines fitted and 28-day forecasts produced successfully.")
"""),
    nbf.v4.new_markdown_cell("### 2. Fit & Forecast Holt-Winters & SARIMA"),
    nbf.v4.new_code_cell("""hw = ExponentialSmoothingForecaster(model_type="holt_winters", seasonal_periods=7).fit(s1i1_train)
pred_hw = hw.predict(horizon=28)
print(f"Holt-Winters execution time: {hw.execution_time_:.3f}s")
print(pred_hw.head(7))
""")
]

nb.cells = cells
with open(notebooks_dir / "04_statistical_forecasting.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Created 04_statistical_forecasting.ipynb")
