"""Unit tests for forecasting models and serialization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronoscast.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from chronoscast.models.ml_forecaster import LightGBMForecaster
from chronoscast.models.statistical import ExponentialSmoothingForecaster


@pytest.fixture
def mini_train_df() -> pd.DataFrame:
    """Fixture with 30 days of data for store 1 item 1."""
    dates = pd.date_range("2016-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store": 1,
        "item": 1,
        "sales": np.random.uniform(10, 50, size=30).round(),
    })


def test_naive_forecaster(mini_train_df: pd.DataFrame):
    model = NaiveForecaster()
    model.fit(mini_train_df)
    preds = model.predict(horizon=7)
    assert len(preds) == 7
    assert "prediction" in preds.columns
    assert (preds["prediction"] >= 0).all()
    # Should all equal the last observation
    last_val = mini_train_df.iloc[-1]["sales"]
    assert (preds["prediction"] == last_val).all()


def test_seasonal_naive_forecaster(mini_train_df: pd.DataFrame):
    model = SeasonalNaiveForecaster(season_length=7)
    model.fit(mini_train_df)
    preds = model.predict(horizon=14)
    assert len(preds) == 14
    assert (preds["prediction"] >= 0).all()


def test_moving_average_forecaster(mini_train_df: pd.DataFrame):
    model = MovingAverageForecaster(window=7)
    model.fit(mini_train_df)
    preds = model.predict(horizon=7)
    assert len(preds) == 7
    expected = mini_train_df["sales"].iloc[-7:].mean()
    assert np.isclose(preds["prediction"].iloc[0], expected)


def test_exponential_smoothing_and_save_load(mini_train_df: pd.DataFrame, tmp_path: Path):
    model = ExponentialSmoothingForecaster(model_type="ses")
    model.fit(mini_train_df)
    preds = model.predict(horizon=5)
    assert len(preds) == 5

    # Test serialization
    save_path = tmp_path / "ses_model.joblib"
    model.save(save_path)
    loaded_model = ExponentialSmoothingForecaster.load(save_path)
    preds_loaded = loaded_model.predict(horizon=5)
    pd.testing.assert_frame_equal(preds, preds_loaded)


def test_lightgbm_forecaster(tmp_path: Path):
    X = pd.DataFrame({
        "store": [1, 1, 1, 1, 1] * 10,
        "item": [1, 1, 1, 1, 1] * 10,
        "day_of_week": [0, 1, 2, 3, 4] * 10,
        "sales_lag_1": np.linspace(10, 50, 50),
        "rolling_mean_7": np.linspace(10, 48, 50),
    })
    y = pd.Series(np.linspace(12, 52, 50))
    model = LightGBMForecaster(n_estimators=10)
    model.fit(X, y)
    assert model.is_fitted is True

    preds = model.predict_features(X)
    assert len(preds) == 50
    assert (preds >= 0).all()
