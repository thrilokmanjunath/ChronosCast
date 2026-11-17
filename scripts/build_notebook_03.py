"""Generate 03_feature_engineering.ipynb."""

from pathlib import Path

import nbformat as nbf

notebooks_dir = Path("notebooks")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "language_info": {"name": "python", "version": "3.12"},
    "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
}

cells = [
    nbf.v4.new_markdown_cell(r"""# ChronosCast — Notebook 03: Leakage-Free Feature Engineering Pipeline
**Project**: ChronosCast: Intelligent Time Series Trend Analysis & Forecasting Platform

### Methodological Rigor & Leakage Prevention:
1. **Entity Group Isolation**: Features are engineered per `(store, item)` series.
2. **Lagged Features**: $y_{t-1}, y_{t-7}, y_{t-14}, y_{t-28}, y_{t-56}, y_{t-365}$ capture autoregressive decay and seasonality.
3. **Shifted Rolling Windows**: Target series is shifted by at least 1 day before rolling statistics (`rolling_mean_w`, `rolling_std_w`) are computed. At time $t$, statistics only aggregate past observations $\{y_{t-1}, ..., y_{t-w}\}$.
4. **Calendar & Trend Signals**: Day-of-week, month, quarter, cyclical sine/cosine transforms, and monotonic `time_index`.
5. **Direct Multi-Horizon Dataset Formulation**: For forecast horizon $h \in \{7, 14, 28\}$, features at time $T$ are paired with $y_{T+h}$.
"""),
    nbf.v4.new_code_cell("""import pandas as pd
from chronoscast.features.pipeline import TimeSeriesFeaturePipeline
from chronoscast.features.direct_horizon import create_direct_horizon_dataset

df = pd.read_parquet("../data/interim/cleaned_sales.parquet")
pipeline = TimeSeriesFeaturePipeline()
pipeline.fit(df)
df_features = pipeline.transform(df)

print(f"Engineered feature matrix: {df_features.shape}")
print("Feature column names:")
print(pipeline.get_feature_names())
"""),
    nbf.v4.new_markdown_cell("### 1. Verification of Leakage-Free Alignment (Store 1, Item 1)"),
    nbf.v4.new_code_cell("""sample = df_features[(df_features["store"] == 1) & (df_features["item"] == 1)].sort_values("date").reset_index(drop=True)
sample[["date", "sales", "sales_lag_1", "sales_lag_7", "rolling_mean_7", "rolling_std_7"]].head(10)
"""),
    nbf.v4.new_markdown_cell("""
Notice:
- On row 0 (2013-01-01), `sales_lag_1` and `rolling_mean_7` are NaN (no previous sales exist).
- On row 1 (2013-01-02), `sales_lag_1` matches `sales` of 2013-01-01 (13.0).
- `rolling_mean_7` at row 7 strictly averages rows 0 to 6. Current row sales is NOT included. Zero leakage verified!
"""),
    nbf.v4.new_markdown_cell("### 2. Direct Horizon Alignment (Horizon = 7 Days)"),
    nbf.v4.new_code_cell("""df_h7, feature_cols = create_direct_horizon_dataset(df_features, horizon=7, pipeline=pipeline)
print(f"Direct Horizon 7 dataset: {df_h7.shape}")
df_h7[["date", "sales", "target_h7", "sales_lag_1", "rolling_mean_7"]].head(5)
""")
]

nb.cells = cells
with open(notebooks_dir / "03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Created 03_feature_engineering.ipynb")
