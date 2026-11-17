"""Time series diagnostics module for ChronosCast.

Implements stationarity testing (ADF, KPSS), autocorrelation (ACF, PACF),
classical decomposition, and outlier diagnostics.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_stationarity_tests(series: pd.Series, max_lags: int | None = None) -> Dict[str, Any]:
    """Execute ADF and KPSS tests on a univariate time series.

    Args:
        series: Time series values with clean index.
        max_lags: Maximum lags for testing.

    Returns:
        Dictionary containing ADF and KPSS test statistics, p-values, critical values,
        and stationary conclusions.
    """
    clean_series = series.dropna()

    # 1. Augmented Dickey-Fuller Test
    # Null hypothesis: Series has a unit root (non-stationary)
    adf_result = adfuller(clean_series, maxlag=max_lags, autolag="AIC")
    adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_crit, adf_icbest = adf_result
    adf_is_stationary = adf_pvalue < 0.05

    # 2. KPSS Test
    # Null hypothesis: Series is trend-stationary / level-stationary
    # statsmodels emits warnings on extreme p-values, which is standard
    try:
        kpss_result = kpss(clean_series, regression="c", nlags="auto")
        kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss_result
        kpss_is_stationary = kpss_pvalue >= 0.05
    except Exception as e:
        logger.warning("KPSS calculation encountered: %s", e)
        kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = np.nan, np.nan, 0, {}
        kpss_is_stationary = False

    # Joint conclusion
    if adf_is_stationary and kpss_is_stationary:
        conclusion = "Stationary (Both ADF and KPSS agree)"
    elif not adf_is_stationary and not kpss_is_stationary:
        conclusion = "Non-Stationary (Both ADF and KPSS agree on unit root / non-stationarity)"
    elif adf_is_stationary and not kpss_is_stationary:
        conclusion = "Difference Stationary (ADF rejects unit root, KPSS rejects level stationarity)"
    else:
        conclusion = "Trend Stationary (ADF does not reject unit root, KPSS does not reject stationarity)"

    return {
        "adf": {
            "statistic": float(np.round(adf_stat, 4)),
            "p_value": float(np.round(adf_pvalue, 6)),
            "lags_used": int(adf_usedlag),
            "n_obs": int(adf_nobs),
            "critical_values": {k: float(np.round(v, 4)) for k, v in adf_crit.items()},
            "is_stationary": bool(adf_is_stationary),
        },
        "kpss": {
            "statistic": float(np.round(kpss_stat, 4)) if not np.isnan(kpss_stat) else None,
            "p_value": float(np.round(kpss_pvalue, 6)) if not np.isnan(kpss_pvalue) else None,
            "lags_used": int(kpss_lags),
            "critical_values": {k: float(np.round(v, 4)) for k, v in kpss_crit.items()},
            "is_stationary": bool(kpss_is_stationary),
        },
        "conclusion": conclusion,
    }


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 35,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute Autocorrelation Function (ACF) and Partial Autocorrelation (PACF).

    Args:
        series: Univariate series.
        nlags: Number of lags to compute.
        alpha: Confidence level significance.

    Returns:
        Dictionary with lag values, acf values, pacf values, and confidence intervals.
    """
    clean_series = series.dropna()
    acf_vals, acf_confint = acf(clean_series, nlags=nlags, alpha=alpha)
    pacf_vals, pacf_confint = pacf(clean_series, nlags=nlags, alpha=alpha, method="ywm")

    return {
        "lags": list(range(len(acf_vals))),
        "acf": acf_vals.tolist(),
        "acf_confint_lower": acf_confint[:, 0].tolist(),
        "acf_confint_upper": acf_confint[:, 1].tolist(),
        "pacf": pacf_vals.tolist(),
        "pacf_confint_lower": pacf_confint[:, 0].tolist(),
        "pacf_confint_upper": pacf_confint[:, 1].tolist(),
    }


def perform_decomposition(
    series: pd.Series,
    model: str = "additive",
    period: int = 7,
) -> Dict[str, Any]:
    """Perform classical seasonal decomposition (additive or multiplicative).

    Args:
        series: Univariate series with regular frequency or explicit date index.
        model: 'additive' or 'multiplicative'.
        period: Seasonal period (default 7 for weekly).

    Returns:
        Dictionary containing dates, observed, trend, seasonal, and resid series.
    """
    clean = series.dropna()
    if model == "multiplicative" and (clean <= 0).any():
        logger.warning("Series contains values <= 0; shifting minimum to 1 for multiplicative decomposition.")
        clean = clean + (1.0 - clean.min())

    result = seasonal_decompose(clean, model=model, period=period, extrapolate_trend="period")

    return {
        "model": model,
        "period": period,
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
        "observed": result.observed,
    }


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Detect outlier boolean mask using Interquartile Range (IQR)."""
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - multiplier * iqr
    upper_bound = q75 + multiplier * iqr
    return (series < lower_bound) | (series > upper_bound)
