# ChronosCast — Model Documentation

## 1. Benchmarked Model Hierarchy

### Level 1: Statistical Baselines
1. **Naive (Persistence)**
   - Formula: $\hat{y}_{t+h} = y_t$
   - Serves as the minimal sanity benchmark. All production models must comfortably beat Naive.
2. **Seasonal Naive**
   - Formula: $\hat{y}_{t+h} = y_{t+h - 7 \lceil h/7 \rceil}$
   - Replicates the matching day of week from the most recent historical week.
3. **Moving Average**
   - Formula: $\hat{y}_{t+h} = \frac{1}{w} \sum_{i=0}^{w-1} y_{t-i}$ with $w = 7$.
   - Captures local recent level while dampening single-day noise.

### Level 2: Classical Statistical Models
4. **Simple Exponential Smoothing (SES)**
   - Suitable for level-stationary series without trend or seasonality.
5. **Holt's Linear Trend**
   - Incorporates recursive trend estimation.
6. **Holt-Winters Seasonal Exponential Smoothing**
   - Additive level, trend, and weekly seasonal components ($s = 7$).
7. **SARIMA (Seasonal Autoregressive Integrated Moving Average)**
   - Fits orders $(p, d, q) \times (P, D, Q)_s$.

### Level 3: Machine Learning Models
8. **Random Forest Regressor**
   - Non-linear ensemble of decision trees trained on engineered lag, rolling, and temporal features.
9. **XGBoost Regressor**
   - Scalable gradient boosted decision trees with depth-wise tree growth.
10. **LightGBM Regressor (Primary Production Model)**
    - Leaf-wise histogram-based gradient boosting.
    - Global pooled architecture: A single unified model trains across all 500 store-item time series simultaneously, capturing shared macro trends and entity-level fixed effects.

## 2. Official Validation Leaderboard

Evaluated strictly on held-out **H1 2017 Validation Window (28-Day Horizon)**:

| Rank | Model | Category | WAPE (%) | MAE | RMSE | sMAPE (%) | Runtime |
|---|---|---|---|---|---|---|---|
| **1** | **LightGBM Regressor (Primary)** | Machine Learning | **10.53%** | **5.9182** | **7.6941** | **12.21%** | 6.54s |
| **2** | **XGBoost Regressor** | Machine Learning | **10.71%** | **6.0178** | **7.8291** | **12.38%** | 0.36s |
| **3** | **Random Forest Regressor** | Machine Learning | **11.13%** | **6.2536** | **8.1763** | **12.79%** | 2.61s |
| **4** | **Holt-Winters** | Statistical | **17.32%** | **5.6228** | **7.3060** | **15.20%** | 0.82s |
| **5** | **Moving Average (w=7)** | Baseline | **17.41%** | **6.9543** | **9.0237** | **14.88%** | 0.09s |
| **6** | **Seasonal Naive (s=7)** | Baseline | **18.23%** | **7.2800** | **9.4009** | **15.42%** | 0.08s |
| **7** | **Naive (Persistence)** | Baseline | **26.58%** | **10.6144** | **13.7986** | **22.14%** | 0.09s |

## 3. Final Test Set Evaluation (H2 2017 — Untouched Until Model Selection)

- **Test Period**: 2017-07-01 to 2017-12-31 (92,000 observations)
- **Test WAPE**: **10.16%**
- **Test MAE**: **6.2391 units**
- **Test RMSE**: **8.0964 units**
- **Test sMAPE**: **11.75%**

## 4. Uncertainty Estimation (Conformal Residual Quantiles)

Prediction intervals are constructed using non-parametric conformal calibration on validation absolute errors:
$$\text{Lower} = \max(0, \hat{y} - q_\alpha), \quad \text{Upper} = \hat{y} + q_\alpha$$

### Empirical Test Set Coverage:
- **80% Target**: **77.68%** empirical coverage (Margin: $\pm 9.40$ units)
- **90% Target**: **88.31%** empirical coverage (Margin: $\pm 12.56$ units)
- **95% Target**: **94.08%** empirical coverage (Margin: $\pm 15.58$ units)
