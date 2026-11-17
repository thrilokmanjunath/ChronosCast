"""Unit tests for evaluation metrics, chronological splitting, and conformal uncertainty."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chronoscast.evaluation.backtesting import ChronologicalSplitter
from chronoscast.evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
    weighted_absolute_percentage_error,
)
from chronoscast.forecasting.uncertainty import ConformalIntervalEstimator


def test_metric_calculations():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 36.0])

    mae = mean_absolute_error(y_true, y_pred)
    assert np.isclose(mae, 2.75)

    rmse = root_mean_squared_error(y_true, y_pred)
    assert rmse > mae

    wape = weighted_absolute_percentage_error(y_true, y_pred)
    # sum(|e|) = 2 + 2 + 3 + 4 = 11, sum(y) = 100 -> 11%
    assert np.isclose(wape, 11.0)


def test_metric_zero_handling():
    y_true = np.array([0.0, 0.0, 10.0])
    y_pred = np.array([2.0, 0.0, 10.0])

    # Should not raise ZeroDivisionError
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    assert not np.isnan(mape)
    assert not np.isnan(smape)
    assert not np.isnan(wape)


def test_chronological_splitter():
    dates = pd.date_range("2013-01-01", "2017-12-31", freq="D")
    df = pd.DataFrame({
        "date": dates,
        "sales": np.random.uniform(10, 50, len(dates)),
    })
    splitter = ChronologicalSplitter(
        train_end="2016-12-31",
        val_end="2017-06-30",
        test_end="2017-12-31",
    )
    train, val, test = splitter.split(df)
    assert train["date"].max() == pd.to_datetime("2016-12-31")
    assert val["date"].min() == pd.to_datetime("2017-01-01")
    assert val["date"].max() == pd.to_datetime("2017-06-30")
    assert test["date"].min() == pd.to_datetime("2017-07-01")
    assert test["date"].max() == pd.to_datetime("2017-12-31")
    # No temporal overlap
    assert len(set(train["date"]).intersection(set(val["date"]))) == 0
    assert len(set(val["date"]).intersection(set(test["date"]))) == 0


def test_conformal_calibrator(tmp_path: Path):
    y_true = np.random.normal(50, 10, 500)
    y_pred = y_true + np.random.normal(0, 3, 500)

    calibrator = ConformalIntervalEstimator(confidence_levels=[0.80, 0.90])
    calibrator.calibrate(y_true, y_pred)
    assert calibrator.is_calibrated is True
    assert "80%" in calibrator.quantiles
    assert "90%" in calibrator.quantiles

    test_preds = np.array([40.0, 50.0, 60.0])
    lower, upper = calibrator.predict_intervals(test_preds, confidence_level=0.90)
    assert (lower <= test_preds).all()
    assert (upper >= test_preds).all()
    assert (lower >= 0).all()

    # Serialization test
    cal_file = tmp_path / "test_calibrator.json"
    calibrator.save(cal_file)
    loaded = ConformalIntervalEstimator.load(cal_file)
    assert loaded.quantiles == calibrator.quantiles
