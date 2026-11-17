# ChronosCast: Intelligent Time Series Trend Analysis & Forecasting Platform

[![CI Pipeline](https://github.com/thrilokmanjunath/ChronosCast/actions/workflows/ci.yml/badge.svg)](https://github.com/thrilokmanjunath/ChronosCast/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: 35 passed](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)](tests/)

---

## 1. Project Introduction
**ChronosCast** is a complete, production-oriented Data Science and MLOps platform engineered for retail demand forecasting, multi-series trend analysis, and uncertainty quantification. Built around a global pooled architecture across 500 individual retail time series, ChronosCast combines rigorous temporal validation, conformal prediction intervals, a high-throughput FastAPI REST API, and an interactive 7-page Streamlit analytical dashboard.

---

## 2. Business Problem
Retail supply chains operate under volatile, seasonal consumer demand. Stockouts result in lost revenue and dissatisfied customers, while excess inventory ties up working capital and causes spoilage or markdowns. Traditional forecasting in retail frequently suffers from:
1. **Target Leakage**: Future observations contaminating rolling statistics, producing artificially inflated backtest scores that fail in production.
2. **Lack of Uncertainty Bounds**: Point forecasts provide no measure of risk or volatility for buffer sizing.
3. **Disconnected Models**: Siloed models that fail to share strength across items and stores.
4. **Non-Scalable Baselines**: Inability to benchmark complex ML models against simple moving averages and seasonal baselines.

---

## 3. Project Objectives
- Ingest and validate 913,000 historical daily observations across 10 stores and 50 SKUs.
- Build a 100% leakage-free feature engineering pipeline with autoregressive lags and shifted rolling windows.
- Benchmark 7 forecasting models across 3 tiers (Baselines, Classical Statistical, Machine Learning).
- Enforce strict chronological validation (2013-2016 Train, H1 2017 Validation, H2 2017 untouched Final Test).
- Calibrate non-parametric conformal prediction intervals (80%, 90%, 95%) and verify empirical coverage on held-out test data.
- Serve dynamic predictions via FastAPI and an interactive Streamlit workspace with prescriptive replenishment planning indicators.

---

## 4. Dataset Source & Attribution
- **Dataset**: Kaggle Store Item Demand Forecasting Challenge (`demand-forecasting-kernels-only`)
- **Official URL**: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
- **Format**: CSV (`train.csv`), 17 MB.
- **License**: Public competition dataset for benchmarking and research. Raw data is excluded from git commits per repository policy.

---

## 5. Dataset Description
- **Total Records**: 913,000 daily observations.
- **Temporal Coverage**: 5 complete calendar years (2013-01-01 through 2017-12-31, 1,826 days).
- **Entities**: 10 stores $\times$ 50 items = 500 parallel time series.
- **Completeness**: 100% full temporal grid ($10 \times 50 \times 1,826 = 913,000$). Zero missing dates, zero negative values.
- **Target Variable**: `sales` (Range: 0 to 231 units, Mean: 52.25, Std: 28.80).

---

## 6. Architecture Overview
```mermaid
flowchart TD
    subgraph Data Pipeline
        Raw[data/raw/train.csv] --> Validator[DataQualityValidator]
        Validator --> Interim[data/interim/cleaned_sales.parquet]
        Interim --> FeaturePipe[TimeSeriesFeaturePipeline]
        FeaturePipe --> Processed[data/processed/features_sales.parquet]
    end

    subgraph Chronological Split
        Processed --> Splitter[ChronologicalSplitter]
        Splitter --> Train[Train: 2013-2016 (730.5k rows)]
        Splitter --> Val[Validation: H1 2017 (90.5k rows)]
        Splitter --> Test[Final Test: H2 2017 (92k rows)]
    end

    subgraph Benchmarking Suite
        Train --> Baselines[Naive / Seasonal Naive / Moving Avg]
        Train --> Statistical[Holt-Winters / SARIMA]
        Train --> ML[LightGBM / XGBoost / Random Forest]
        Val --> Leaderboard[Leaderboard Benchmarking]
        Val --> Conformal[Conformal Residual Calibration]
    end

    subgraph Deployment
        ML --> Engine[ForecastingEngine]
        Conformal --> Engine
        Engine --> API[FastAPI :8000]
        Engine --> UI[Streamlit UI :8501]
    end
```

---

## 7. Technology Stack
- **Language**: Python 3.12 (CPython)
- **Package Management**: Astral `uv`
- **Data Engineering**: Pandas 2.2+, NumPy 1.26+, PyArrow
- **Statistical Modeling**: Statsmodels 0.14+ (ADF, KPSS, ACF, PACF, Holt-Winters, SARIMAX)
- **Machine Learning**: LightGBM 4.3+, XGBoost 2.0+, Scikit-learn 1.4+
- **Experiment Tracking**: MLflow 2.11+
- **Serving / API**: FastAPI 0.110+, Uvicorn, Pydantic v2
- **Frontend / Dashboard**: Streamlit 1.32+, Plotly 5.19+
- **Quality Assurance**: Pytest 8.1+, Pytest-Cov, Ruff 0.3+
- **DevOps & Containers**: Docker, Docker Compose, GitHub Actions CI

---

## 8. Installation

### Quickstart with `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/thrilokmanjunath/ChronosCast.git
cd ChronosCast

# Create virtual environment with Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install all dependencies and development tools
uv pip install -e ".[dev]"
```

### Standard `pip` Alternative
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 9. Dataset Acquisition
The dataset is acquired reproducibly via script:
```bash
python scripts/download_data.py
```
*Note: For Kaggle API users, place `~/.kaggle/kaggle.json` and run `kaggle competitions download -c demand-forecasting-kernels-only -p data/raw/`.*

---

## 10. Data Pipeline
Execute the end-to-end data preparation workflow:
```bash
# 1. Clean, validate, and serialize interim dataset
python src/chronoscast/data/cleaner.py

# 2. Engineer leakage-free features across all 500 series
python scripts/build_features.py
```
- Raw data in `data/raw/train.csv` is preserved without mutation.
- Verified interim dataset stored as `data/interim/cleaned_sales.parquet`.
- Enriched feature matrix cached in `data/processed/features_sales.parquet`.

---

## 11. Exploratory Data Analysis
Generate all analytical figures and diagnostic reports:
```bash
python scripts/generate_eda_artifacts.py
```
### Key Empirical Findings:
- **Stationarity**: Augmented Dickey-Fuller (ADF) rejects unit root ($p < 0.001$), while KPSS rejects level stationarity ($p < 0.01$). First-differencing achieves full trend-stationarity.
- **Weekly Rhythm**: Fridays, Saturdays, and Sundays exhibit 45–55% higher sales than Mondays and Tuesdays.
- **Annual Seasonality**: Peak demand occurs in July (summer peak), with an annual trough in January.
- **Inter-Store Variance**: Store 2 and Store 8 generate the highest volume (each ~12.8% network share), while Store 7 generates the lowest (~7.6%).

---

## 12. Feature Engineering Pipeline
The `TimeSeriesFeaturePipeline` engineers 27 features with zero future target leakage:
- **Autoregressive Lags**: `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_28`, `sales_lag_56`, `sales_lag_365`.
- **Shifted Rolling Windows**: `rolling_mean_{7,14,28,56}` and `rolling_std_{7,14,28}` computed on $y_{t-1}$.
- **Calendar & Deterministic Trend**: `day`, `day_of_week`, `week`, `month`, `quarter`, `year`, `is_weekend`, Fourier terms (`sin`/`cos`), and `time_index`.

---

## 13. Model Descriptions
1. **Baselines**:
   - `NaiveForecaster`: Last observed value.
   - `SeasonalNaiveForecaster`: Matching day from previous week ($s=7$).
   - `MovingAverageForecaster`: 7-day trailing average.
2. **Statistical**:
   - `ExponentialSmoothingForecaster`: Holt-Winters additive trend and seasonality.
   - `ARIMAForecaster`: Seasonal ARIMA.
3. **Machine Learning**:
   - `LightGBMForecaster` (**Primary**): Global pooled model trained across all 500 series with early stopping.
   - `RandomForestForecaster`: Random forest ensemble.
   - `XGBoostForecaster`: Gradient boosted trees.

---

## 14. Validation Strategy
- **Train Period**: `2013-01-01` to `2016-12-31` (730,500 observations)
- **Validation Period**: `2017-01-01` to `2017-06-30` (90,500 observations)
- **Final Test Period**: `2017-07-01` to `2017-12-31` (92,000 observations)
*The final test set remained untouched until model selection was locked.*

---

## 15. Evaluation Metrics
- **MAE**: Mean Absolute Error $\frac{1}{N} \sum |y - \hat{y}|$
- **RMSE**: Root Mean Squared Error $\sqrt{\frac{1}{N} \sum (y - \hat{y})^2}$
- **MAPE**: Mean Absolute Percentage Error with $\epsilon$-floor zero protection.
- **sMAPE**: Symmetric Mean Absolute Percentage Error (bounded 0–200%).
- **WAPE**: Weighted Absolute Percentage Error $\frac{\sum |y - \hat{y}|}{\sum y} \times 100\%$ (primary retail accuracy metric).

---

## 16. Actual Experimental Results

### Official Validation Benchmark Leaderboard (H1 2017 — 28-Day Window)
| Rank | Model Name | Category | WAPE (%) | MAE | RMSE | sMAPE (%) | Runtime |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| **1** | **LightGBM Regressor (Primary)** | **Machine Learning** | **10.53%** | **5.9182** | **7.6941** | **12.21%** | **6.54s** |
| **2** | XGBoost Regressor | Machine Learning | 10.71% | 6.0178 | 7.8291 | 12.38% | 0.36s |
| **3** | Random Forest Regressor | Machine Learning | 11.13% | 6.2536 | 8.1763 | 12.79% | 2.61s |
| **4** | Holt-Winters (Sample Series) | Statistical | 17.32% | 5.6228 | 7.3060 | 15.20% | 0.82s |
| **5** | Moving Average (w=7) | Baseline | 17.41% | 6.9543 | 9.0237 | 14.88% | 0.09s |
| **6** | Seasonal Naive (s=7) | Baseline | 18.23% | 7.2800 | 9.4009 | 15.42% | 0.08s |
| **7** | Naive (Persistence) | Baseline | 26.58% | 10.6144 | 13.7986 | 22.14% | 0.09s |

### Final Untouched Test Set Performance (H2 2017 — LightGBM)
- **WAPE**: **10.16%**
- **MAE**: **6.2391 units**
- **RMSE**: **8.0964 units**
- **sMAPE**: **11.75%**

### Conformal Prediction Interval Coverage (Held-out Test Verification)
- **80% Nominal Target** $\rightarrow$ **77.68% Empirical Coverage** ($\pm 9.40$ units)
- **90% Nominal Target** $\rightarrow$ **88.31% Empirical Coverage** ($\pm 12.56$ units)
- **95% Nominal Target** $\rightarrow$ **94.08% Empirical Coverage** ($\pm 15.58$ units)

---

## 17. Dashboard & Visualizations
The Streamlit application provides 7 interactive analytical pages + inventory planning:
- **Executive Overview**: High-level network demand and active model KPIs.
- **Data Explorer**: Granular filtering across stores, products, and custom date ranges.
- **Trend Analysis**: Decomposition, seasonal heatmaps, rolling volatility, and ACF/PACF.
- **Forecast Studio**: Interactive out-of-sample forecast generation with shaded conformal uncertainty bands.
- **Model Comparison**: Interactive validation leaderboard and runtime benchmarks.
- **Forecast Evaluation**: Error breakdown by store, item velocity tier, and feature importances.
- **Inventory Insights**: Dynamic safety stock, reorder point (ROP), lead-time demand, and stockout risk indicators.
- **Reports**: Instant export of all benchmark metrics and data quality audits.

Figures are saved in `reports/figures/` (PNG and interactive HTML).

---

## 18. API Documentation
Run the FastAPI backend:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Redoc Documentation: `http://localhost:8000/redoc`

### Key Endpoints:
- `GET /health`: Platform operational health.
- `GET /api/v1/datasets/summary`: Overview of 913,000 observations.
- `GET /api/v1/stores`: Aggregated sales statistics across 10 stores.
- `GET /api/v1/items`: Demand velocity metrics across 50 items.
- `POST /api/v1/forecast`: Dynamic forecast generation with lower/upper prediction bounds.
- `GET /api/v1/model/performance`: Official leaderboard and test evaluation metrics.

---

## 19. Docker Execution
Run the entire platform via Docker Compose:
```bash
# Build and launch both API (:8000) and Dashboard (:8501)
docker compose up --build
```
- API Docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

---

## 20. Testing & Quality Assurance
Run the complete automated test suite:
```bash
# Run 35 automated unit and integration tests
pytest tests/ -v

# Run test coverage audit
pytest tests/ --cov=chronoscast --cov-report=term-missing

# Run Ruff linter and style check
ruff check src/ api/ app/ tests/ scripts/
```
All **35 tests pass** with zero errors across:
- Data loading and schema validation
- Time series diagnostics and decomposition
- Autoregressive feature alignment and zero-leakage tests
- Baseline and ML model fit/predict
- Evaluation metrics and zero handling
- Conformal prediction interval coverage
- FastAPI REST endpoints
- Streamlit application syntax and compilability

---

## 21. Known Limitations
1. **Absence of Exogenous Factors**: The dataset contains only sales volume and dates; pricing, promotional calendars, and weather events are unavailable.
2. **Realized Sales vs True Demand**: Out-of-stock events in the historical data truncate realized demand, which can lead to conservative estimates during peak demand.
3. **Hypothetical Inventory Modeling**: Inventory parameters (lead times, holding costs) are planning indicators and must be mapped to enterprise ERP data in production.

---

## 22. Future Engineering Improvements
- **Deep Learning Architectures**: Benchmark Temporal Fusion Transformers (TFT) and N-BEATS using PyTorch Forecasting.
- **Hierarchical Reconciliation**: Implement MinT (Minimum Trace) reconciliation for mathematical consistency between SKU, store, and network forecasts.
- **Automated Hyperparameter Optimization**: Integrate Optuna for distributed bayesian tuning.
- **Streaming Ingestion**: Wire Apache Kafka to stream daily transactions into the feature store.

---

## 23. License & Author
- **Author**: Thrilok Manjunath ([@thrilokmanjunath](https://github.com/thrilokmanjunath))
- **Repository**: [ChronosCast](https://github.com/thrilokmanjunath/ChronosCast)
