# Dataset Documentation: Store Item Demand Forecasting

## 1. Overview
This dataset contains historical daily sales records across 10 different retail stores and 50 unique items over a 5-year period (2013-01-01 through 2017-12-31).

- **Competition / Origin**: Kaggle Store Item Demand Forecasting Challenge (`demand-forecasting-kernels-only`)
- **Official URL**: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
- **Total Observations**: 913,000 daily sales records
- **Number of Stores**: 10
- **Number of Items**: 50
- **Time Horizon**: 5 years (1,826 calendar days per series, including leap year 2016)
- **Total Time Series**: 500 individual store-item time series ($10 \times 50$)
- **Completeness**: 100% full temporal grid ($500 \times 1,826 = 913,000$ rows) without missing date gaps.

## 2. Schema Specification

| Column Name | Data Type | Description | Valid Range | Example |
|---|---|---|---|---|
| `date` | `string` / `datetime64[ns]` | Transaction calendar date (YYYY-MM-DD) | 2013-01-01 to 2017-12-31 | `2013-01-01` |
| `store` | `int64` / categorical | Unique store identifier | 1 through 10 | `1` |
| `item` | `int64` / categorical | Unique item/SKU identifier | 1 through 50 | `1` |
| `sales` | `int64` / `float64` | Units sold on the specified date | Non-negative integer (0 to 231) | `13` |

## 3. Data Acquisition & Reproducibility Procedure

### Automatic / Scripted Acquisition
The project automated fetch downloads the verifiable raw data directly into `data/raw/train.csv`:
```bash
python scripts/download_data.py
```

### Manual Download Procedure (via Kaggle API or Browser)
1. Install the official Kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Place your Kaggle API key at `~/.kaggle/kaggle.json`.
3. Download the competition archive:
   ```bash
   kaggle competitions download -c demand-forecasting-kernels-only -p data/raw/
   unzip data/raw/demand-forecasting-kernels-only.zip -d data/raw/
   ```
4. Verify checksum and row count:
   ```bash
   wc -l data/raw/train.csv  # Expected: 913001 lines (header + 913,000 records)
   ```

## 4. Characteristics and Inherent Patterns
- **Weekly Seasonality**: Clear surge in sales on Fridays, Saturdays, and Sundays compared to weekdays.
- **Annual Seasonality**: High demand during summer months (June–August) and holiday periods; low trough in January–February.
- **Multi-Year Trend**: Steady upward sales growth year-over-year from 2013 to 2017 across all stores and items.
- **Inter-Store Variability**: Store sizes/locations create consistent scale offsets (e.g., Store 2 and Store 8 exhibit higher volume than Store 7).
- **Inter-Item Heterogeneity**: Different baseline velocities across the 50 items (fast-moving staples vs. niche slower-moving items).

## 5. Data Limitations and Realistic Assumptions
- **Absence of Exogenous Covariates**: No pricing, markdown, promotional calendar, local weather, or competitor data is provided.
- **No Stockout / Inventory Signals**: Recorded sales reflect truncated realized demand if out-of-stock events occurred.
- **Aggregated Demand**: Transactions are aggregated at daily level without hourly or basket-level metadata.

## 6. License and Usage
The dataset was published under public competition terms for educational and benchmarking purposes. ChronosCast adheres strictly to licensing restrictions: raw data is not committed to git repositories, and processing pipelines ensure full reproducibility from raw source.
