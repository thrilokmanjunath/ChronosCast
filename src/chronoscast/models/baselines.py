"""Statistical baseline forecasting models for ChronosCast."""

from __future__ import annotations

import time
from typing import Any, Dict

import pandas as pd

from chronoscast.models.base import BaseForecaster
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class NaiveForecaster(BaseForecaster):
    """Naive persistence forecast: y_{t+h} = y_t (last observed value)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="naive", **kwargs)
        self.last_observations: Dict[tuple, float] = {}
        self.last_dates: Dict[tuple, pd.Timestamp] = {}

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> NaiveForecaster:
        t0 = time.time()
        df_sorted = train_df.sort_values(by=["store", "item", "date"])
        # Extract last observation per entity
        last_rows = df_sorted.groupby(["store", "item"], observed=True).last().reset_index()
        for _, row in last_rows.iterrows():
            key = (int(row["store"]), int(row["item"]))
            self.last_observations[key] = float(row["sales"])
            self.last_dates[key] = pd.to_datetime(row["date"])

        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        records = []
        for (store, item), last_val in self.last_observations.items():
            start_date = self.last_dates[(store, item)]
            future_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            for d in future_dates:
                records.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    "prediction": max(0.0, last_val),
                })
        return pd.DataFrame(records)


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive forecast: y_{t+h} = y_{t+h - season_length * ceil(h/season_length)}."""

    def __init__(self, season_length: int = 7, **kwargs: Any) -> None:
        super().__init__(name="seasonal_naive", season_length=season_length, **kwargs)
        self.season_length = season_length
        self.history: Dict[tuple, pd.Series] = {}
        self.last_dates: Dict[tuple, pd.Timestamp] = {}

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> SeasonalNaiveForecaster:
        t0 = time.time()
        df_sorted = train_df.sort_values(by=["store", "item", "date"])
        for (store, item), group in df_sorted.groupby(["store", "item"], observed=True):
            key = (int(store), int(item))
            self.history[key] = group["sales"].iloc[-self.season_length:].values
            self.last_dates[key] = pd.to_datetime(group["date"].iloc[-1])

        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        records = []
        for (store, item), hist in self.history.items():
            start_date = self.last_dates[(store, item)]
            future_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            for step_idx, d in enumerate(future_dates):
                # Map step_idx to matching day of season
                seasonal_idx = step_idx % self.season_length
                pred = hist[seasonal_idx] if seasonal_idx < len(hist) else hist[-1]
                records.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    "prediction": max(0.0, float(pred)),
                })
        return pd.DataFrame(records)


class MovingAverageForecaster(BaseForecaster):
    """Moving Average forecast: average of last window observations."""

    def __init__(self, window: int = 7, **kwargs: Any) -> None:
        super().__init__(name="moving_average", window=window, **kwargs)
        self.window = window
        self.means: Dict[tuple, float] = {}
        self.last_dates: Dict[tuple, pd.Timestamp] = {}

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> MovingAverageForecaster:
        t0 = time.time()
        df_sorted = train_df.sort_values(by=["store", "item", "date"])
        for (store, item), group in df_sorted.groupby(["store", "item"], observed=True):
            key = (int(store), int(item))
            recent = group["sales"].iloc[-self.window:]
            self.means[key] = float(recent.mean())
            self.last_dates[key] = pd.to_datetime(group["date"].iloc[-1])

        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        records = []
        for (store, item), mean_val in self.means.items():
            start_date = self.last_dates[(store, item)]
            future_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=horizon, freq="D")
            for d in future_dates:
                records.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    "prediction": max(0.0, mean_val),
                })
        return pd.DataFrame(records)
