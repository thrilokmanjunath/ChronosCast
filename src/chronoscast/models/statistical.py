"""Classical statistical forecasting models for ChronosCast.

Wraps statsmodels implementations of:
- Simple Exponential Smoothing (SES)
- Holt's Linear Trend
- Holt-Winters Seasonal Exponential Smoothing
- ARIMA & SARIMA
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from chronoscast.models.base import BaseForecaster
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential smoothing forecaster supporting SES, Holt, and Holt-Winters."""

    def __init__(
        self,
        model_type: str = "holt_winters",  # 'ses', 'holt', 'holt_winters'
        seasonal_periods: int = 7,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=f"exp_smoothing_{model_type}", model_type=model_type, **kwargs)
        self.model_type = model_type
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.fitted_models: Dict[tuple, Any] = {}
        self.last_dates: Dict[tuple, pd.Timestamp] = {}

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> ExponentialSmoothingForecaster:
        t0 = time.time()
        df_sorted = train_df.sort_values(by=["store", "item", "date"])

        for (store, item), group in df_sorted.groupby(["store", "item"], observed=True):
            key = (int(store), int(item))
            series = group.set_index("date")["sales"].asfreq("D")
            self.last_dates[key] = group["date"].iloc[-1]

            try:
                if self.model_type == "ses":
                    model = SimpleExpSmoothing(series, initialization_method="estimated").fit()
                elif self.model_type == "holt":
                    model = Holt(series, initialization_method="estimated").fit()
                else:  # holt_winters
                    model = ExponentialSmoothing(
                        series,
                        trend=self.trend,
                        seasonal=self.seasonal,
                        seasonal_periods=self.seasonal_periods,
                        initialization_method="estimated",
                    ).fit()
                self.fitted_models[key] = model
            except Exception as e:
                logger.warning("Fit error for series %s in %s: %s. Falling back to simple mean.", key, self.name, e)
                self.fitted_models[key] = float(series.mean())

        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        records = []
        for (store, item), model in self.fitted_models.items():
            start_date = self.last_dates[(store, item)]
            future_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=horizon, freq="D")

            if hasattr(model, "forecast"):
                preds = model.forecast(horizon).values
            else:
                preds = np.repeat(float(model), horizon)

            for d, pred_val in zip(future_dates, preds, strict=False):
                records.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    "prediction": max(0.0, float(pred_val)),
                })
        return pd.DataFrame(records)


class ARIMAForecaster(BaseForecaster):
    """ARIMA and SARIMA forecaster via statsmodels SARIMAX."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        is_seasonal: bool = True,
        **kwargs: Any,
    ) -> None:
        name = "sarima" if is_seasonal else "arima"
        super().__init__(name=name, order=order, seasonal_order=seasonal_order, is_seasonal=is_seasonal, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order if is_seasonal else (0, 0, 0, 0)
        self.is_seasonal = is_seasonal
        self.fitted_models: Dict[tuple, Any] = {}
        self.last_dates: Dict[tuple, pd.Timestamp] = {}

    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> ARIMAForecaster:
        t0 = time.time()
        df_sorted = train_df.sort_values(by=["store", "item", "date"])

        for (store, item), group in df_sorted.groupby(["store", "item"], observed=True):
            key = (int(store), int(item))
            series = group.set_index("date")["sales"].asfreq("D")
            self.last_dates[key] = group["date"].iloc[-1]

            try:
                # Use standard low-order SARIMAX with enforce_stationarity=False for robustness
                sarima = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order if self.is_seasonal else None,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_res = sarima.fit(disp=False, maxiter=50)
                self.fitted_models[key] = fit_res
            except Exception as e:
                logger.warning("SARIMAX fit failed for %s: %s. Storing mean fallback.", key, e)
                self.fitted_models[key] = float(series.mean())

        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        return self

    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        records = []
        for (store, item), model in self.fitted_models.items():
            start_date = self.last_dates[(store, item)]
            future_dates = pd.date_range(start_date + pd.Timedelta(days=1), periods=horizon, freq="D")

            if hasattr(model, "forecast"):
                preds = model.forecast(horizon).values
            else:
                preds = np.repeat(float(model), horizon)

            for d, pred_val in zip(future_dates, preds, strict=False):
                records.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    "prediction": max(0.0, float(pred_val)),
                })
        return pd.DataFrame(records)
