# ChronosCast — Data Dictionary

## 1. Raw Dataset Columns (`train.csv`)

| Column | Type | Range | Description |
|---|---|---|---|
| `date` | `string` / `datetime` | `2013-01-01` to `2017-12-31` | Calendar transaction date |
| `store` | `int32` | `1` to `10` | Identifier for retail store location |
| `item` | `int32` | `1` to `50` | Identifier for retail product SKU |
| `sales` | `float32` | `0` to `231` | Realized daily sales count |

## 2. Engineered Feature Columns (`features_sales.parquet`)

### Calendar Features
- `day`: Day of month (1 to 31)
- `day_of_week`: Day of week index (0 = Monday, 6 = Sunday)
- `week`: ISO calendar week (1 to 53)
- `month`: Calendar month (1 to 12)
- `quarter`: Calendar quarter (1 to 4)
- `year`: Calendar year (2013 to 2017)
- `is_weekend`: Binary flag (1 if Saturday or Sunday, else 0)
- `day_of_week_sin` / `day_of_week_cos`: Cyclical Fourier transformation of weekly cycle
- `month_sin` / `month_cos`: Cyclical Fourier transformation of annual cycle

### Deterministic Trend
- `time_index`: Integer count of days elapsed since reference date (`2013-01-01`)

### Autoregressive Lag Features
*Computed strictly per entity `(store, item)`*:
- `sales_lag_1`: Sales at $t - 1$
- `sales_lag_7`: Sales at $t - 7$ (same day previous week)
- `sales_lag_14`: Sales at $t - 14$ (same day 2 weeks ago)
- `sales_lag_28`: Sales at $t - 28$ (same day 4 weeks ago)
- `sales_lag_56`: Sales at $t - 56$ (same day 8 weeks ago)
- `sales_lag_365`: Sales at $t - 365$ (same day 1 year ago)

### Shifted Rolling Window Statistics
*Computed on $y_{t-1}$ to prevent contemporaneous target leakage*:
- `rolling_mean_7`: 7-day moving average over $[t-7, ..., t-1]$
- `rolling_mean_14`: 14-day moving average over $[t-14, ..., t-1]$
- `rolling_mean_28`: 28-day moving average over $[t-28, ..., t-1]$
- `rolling_mean_56`: 56-day moving average over $[t-56, ..., t-1]$
- `rolling_std_7`: 7-day moving standard deviation over $[t-7, ..., t-1]$
- `rolling_std_14`: 14-day moving standard deviation over $[t-14, ..., t-1]$
- `rolling_std_28`: 28-day moving standard deviation over $[t-28, ..., t-1]$
