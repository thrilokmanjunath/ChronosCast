"""Script to generate notebooks 01_data_exploration and 02_time_series_analysis."""

from pathlib import Path

import nbformat as nbf

notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(parents=True, exist_ok=True)

# ----------------- Notebook 1: Data Exploration -----------------
nb1 = nbf.v4.new_notebook()
nb1.metadata = {
    "language_info": {"name": "python", "version": "3.12"},
    "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
}

cells1 = [
    nbf.v4.new_markdown_cell("""# ChronosCast — Notebook 01: Exploratory Data Analysis
**Project**: ChronosCast: Intelligent Time Series Trend Analysis & Forecasting Platform
**Dataset**: Kaggle Store Item Demand Forecasting Challenge (913,000 observations)

### Notebook Scope:
1. Environment and dataset loading
2. Schema inspection, data types, missing value audit, and duplicate checks
3. Univariate and bivariate sales distributions across stores and items
4. Date coverage and calendar verification
"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import plotly.express as px
from chronoscast.data.loader import load_raw_data
from chronoscast.data.validator import DataQualityValidator
from chronoscast.analysis.eda import TimeSeriesEDA

# Load cleaned interim dataset
df = pd.read_parquet("../data/interim/cleaned_sales.parquet")
print(f"Loaded {len(df):,} rows. Shape: {df.shape}")
df.head()
"""),
    nbf.v4.new_markdown_cell("### 1. Data Quality and Summary Statistics"),
    nbf.v4.new_code_cell("""eda = TimeSeriesEDA(df)
summary = eda.overall_summary()
for k, v in summary.items():
    print(f"{k}: {v}")
"""),
    nbf.v4.new_markdown_cell("### 2. Store-Level Volume & Share"),
    nbf.v4.new_code_cell("""store_stats = eda.store_level_summary()
store_stats
"""),
    nbf.v4.new_markdown_cell("### 3. Product Demand Velocity"),
    nbf.v4.new_code_cell("""item_stats = eda.item_level_summary()
print("Top 5 Items by Demand:")
display(item_stats.head(5))
print("Bottom 5 Items by Demand:")
display(item_stats.tail(5))
"""),
    nbf.v4.new_markdown_cell("### 4. Sales Distribution and Summary Metrics"),
    nbf.v4.new_code_cell("""print(df["sales"].describe())
""")
]
nb1.cells = cells1
with open(notebooks_dir / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)
print("Created 01_data_exploration.ipynb")

# ----------------- Notebook 2: Time Series Analysis -----------------
nb2 = nbf.v4.new_notebook()
nb2.metadata = {
    "language_info": {"name": "python", "version": "3.12"},
    "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
}

cells2 = [
    nbf.v4.new_markdown_cell("""# ChronosCast — Notebook 02: Time Series Trend & Seasonality Diagnostics
**Project**: ChronosCast: Intelligent Time Series Trend Analysis & Forecasting Platform

### Notebook Scope:
1. Long-term trend analysis with rolling moving averages and rolling volatility
2. Multi-granularity seasonality: Day of week, month, quarter, annual growth
3. Classical time series decomposition (Observed, Trend, Seasonal, Residual)
4. Stationarity verification using Augmented Dickey-Fuller (ADF) & KPSS tests
5. Autocorrelation Function (ACF) & Partial Autocorrelation Function (PACF)
"""),
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from chronoscast.analysis.eda import TimeSeriesEDA
from chronoscast.analysis.diagnostics import run_stationarity_tests, compute_acf_pacf, perform_decomposition

df = pd.read_parquet("../data/interim/cleaned_sales.parquet")
eda = TimeSeriesEDA(df)
daily = eda.daily_aggregated_sales()
daily.head()
"""),
    nbf.v4.new_markdown_cell("### 1. Stationarity Tests (ADF & KPSS)"),
    nbf.v4.new_code_cell("""# Test on aggregate daily demand
agg_series = daily.set_index("date")["sales"]
diag_agg = run_stationarity_tests(agg_series)
print("Aggregate Daily Demand Stationarity:")
print(f"  ADF Statistic: {diag_agg['adf']['statistic']} (p-value: {diag_agg['adf']['p_value']}) -> Stationary: {diag_agg['adf']['is_stationary']}")
print(f"  KPSS Statistic: {diag_agg['kpss']['statistic']} (p-value: {diag_agg['kpss']['p_value']}) -> Stationary: {diag_agg['kpss']['is_stationary']}")
print(f"  Conclusion: {diag_agg['conclusion']}")
"""),
    nbf.v4.new_markdown_cell("### 2. Time Series Decomposition"),
    nbf.v4.new_code_cell("""decomp = perform_decomposition(agg_series, model="additive", period=7)
print("Decomposition successful.")
print(f"Trend mean: {decomp['trend'].mean():.2f}")
print(f"Seasonal amplitude: {decomp['seasonal'].max() - decomp['seasonal'].min():.2f}")
print(f"Residual std: {decomp['resid'].std():.2f}")
"""),
    nbf.v4.new_markdown_cell("### 3. Autocorrelation & Partial Autocorrelation Diagnostics"),
    nbf.v4.new_code_cell("""s1i1 = df[(df["store"] == 1) & (df["item"] == 1)].sort_values("date").set_index("date")["sales"]
acf_res = compute_acf_pacf(s1i1, nlags=14)
print("Lags 1 to 14 ACF values for Store 1, Item 1:")
for lag, val in zip(acf_res["lags"][1:15], acf_res["acf"][1:15]):
    print(f"  Lag {lag:2d}: {val:.4f}")
""")
]
nb2.cells = cells2
with open(notebooks_dir / "02_time_series_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)
print("Created 02_time_series_analysis.ipynb")
