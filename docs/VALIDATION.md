# ChronosCast — Validation & Leakage Prevention Strategy

## 1. Core Principle: Leakage-Free Temporal Validation
Standard K-Fold cross-validation randomly shuffles observations, which catastrophically violates temporal causality in time series:
- Model would train on future data to predict past data.
- Autoregressive features would cause severe data snooping.
- Reported accuracy would be artificially inflated and fail in live production.

ChronosCast enforces **strict chronological validation**:

```
[=================== TRAIN: 2013-01-01 to 2016-12-31 ===================] [=== VAL: H1 2017 ===] [=== TEST: H2 2017 ===]
                       730,500 observations                             90,500 observations    92,000 observations
```

## 2. Guardrails Against Feature Contamination
1. **Entity Group Isolation**: Features are engineered strictly within each `(store, item)` series. No cross-entity index bleeding.
2. **Pre-Calculation Target Shift**: The target series is shifted by at least 1 day ($y_{t-1}$) prior to computing rolling statistics (`rolling_mean_w`, `rolling_std_w`). At time $t$, statistics aggregate only $\{y_{t-1}, ..., y_{t-w}\}$.
3. **No Target Encoding Contamination**: Categoricals (`store`, `item`) are represented as integer entity indices or one-hot/embedding categoricals without target-leakage out-of-fold statistics.
4. **Untouched Final Test Set**: The test set (2017-07-01 to 2017-12-31) was not touched during model exploration, hyperparameter tuning, or interval calibration. It was evaluated strictly once at the end of the pipeline.

## 3. Rolling-Origin Backtesting Architecture
In addition to the primary 3-way holdout split, ChronosCast provides `RollingOriginBacktester` across multiple historical forecast origins:
- For origin $T$:
  - Train strictly on $t \le T$
  - Forecast $T+1$ through $T+H$ for $H \in \{7, 14, 28\}$ days
  - Compare forecasts with actuals over $[T+1, T+H]$
  - Metrics recorded independently per window and aggregated across origins.
