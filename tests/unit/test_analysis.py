"""Unit tests for time series analysis, EDA, and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronoscast.analysis.diagnostics import (
    compute_acf_pacf,
    perform_decomposition,
    run_stationarity_tests,
)
from chronoscast.analysis.eda import TimeSeriesEDA


@pytest.fixture
def sample_ts_series() -> pd.Series:
    """Generate a synthetic 100-day series with linear trend and weekly seasonality."""
    np.random.seed(42)
    dates = pd.date_range("2017-01-01", periods=100, freq="D")
    trend = np.linspace(10, 30, 100)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 7)
    noise = np.random.normal(0, 1, 100)
    return pd.Series(trend + seasonal + noise, index=dates, name="sales")


def test_eda_summaries():
    df = pd.read_parquet("data/interim/cleaned_sales.parquet")
    eda = TimeSeriesEDA(df)
    summary = eda.overall_summary()
    assert summary["total_observations"] == 913_000
    assert summary["store_count"] == 10
    assert summary["item_count"] == 50
    assert summary["total_days"] == 1826

    stores = eda.store_level_summary()
    assert len(stores) == 10
    assert "sales_share_pct" in stores.columns

    items = eda.item_level_summary()
    assert len(items) == 50


def test_stationarity_tests(sample_ts_series: pd.Series):
    res = run_stationarity_tests(sample_ts_series)
    assert "adf" in res
    assert "kpss" in res
    assert "conclusion" in res
    assert isinstance(res["adf"]["statistic"], float)
    assert isinstance(res["adf"]["p_value"], float)
    assert isinstance(res["adf"]["is_stationary"], bool)


def test_acf_pacf_computation(sample_ts_series: pd.Series):
    res = compute_acf_pacf(sample_ts_series, nlags=14)
    assert len(res["acf"]) == 15  # lag 0 to 14
    assert len(res["pacf"]) == 15
    assert np.isclose(res["acf"][0], 1.0)  # lag 0 autocorrelation is 1


def test_decomposition(sample_ts_series: pd.Series):
    decomp = perform_decomposition(sample_ts_series, model="additive", period=7)
    assert len(decomp["trend"]) == len(sample_ts_series)
    assert len(decomp["seasonal"]) == len(sample_ts_series)
    assert len(decomp["resid"]) == len(sample_ts_series)
