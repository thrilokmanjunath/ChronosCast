"""Production forecasting engine for ChronosCast."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from chronoscast.forecasting.uncertainty import ConformalIntervalEstimator
from chronoscast.models.base import BaseForecaster
from chronoscast.models.ml_forecaster import LightGBMForecaster
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class ForecastingEngine:
    """End-to-end forecasting service supporting individual, store, item, and total aggregate demand."""

    def __init__(
        self,
        model_path: str | Path = "artifacts/models/best_lightgbm.joblib",
        calibrator_path: str | Path = "artifacts/models/uncertainty_calibrator.json",
        features_path: str | Path = "data/processed/features_sales.parquet",
    ) -> None:
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path)
        self.features_path = Path(features_path)

        self.model: Optional[BaseForecaster] = None
        self.calibrator: Optional[ConformalIntervalEstimator] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained model weights and uncertainty calibrator."""
        if self.model_path.exists():
            self.model = LightGBMForecaster.load(self.model_path)
            logger.info("ForecastingEngine loaded model from: %s", self.model_path)
        else:
            logger.warning("Model file not found at %s. Will load on demand if trained.", self.model_path)

        if self.calibrator_path.exists():
            self.calibrator = ConformalIntervalEstimator.load(self.calibrator_path)
            logger.info("ForecastingEngine loaded uncertainty calibrator from: %s", self.calibrator_path)
        else:
            logger.warning("Calibrator file not found at %s.", self.calibrator_path)

    def generate_forecast(
        self,
        store: Optional[int] = None,
        item: Optional[int] = None,
        horizon: int = 7,
        confidence_level: float = 0.90,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate forecasts and prediction intervals.

        Args:
            store: Store ID (1-10) or None for all stores.
            item: Item ID (1-50) or None for all items.
            horizon: Number of days forward (7, 14, 28).
            confidence_level: Confidence level (0.80, 0.90, 0.95).
            model_name: Optional model override.

        Returns:
            Structured forecast payload with dates, point predictions, lower/upper bounds,
            aggregation level, and metadata.
        """
        if horizon not in [7, 14, 28]:
            logger.warning("Horizon %d provided; standard supported horizons are [7, 14, 28].", horizon)

        # Load recent feature observations for inference
        df_feat = pd.read_parquet(self.features_path)
        df_feat["date"] = pd.to_datetime(df_feat["date"])

        # We predict the immediate out-of-sample period following the dataset's final date
        last_date = df_feat["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        # Filter entities
        stores = [store] if store is not None else sorted(df_feat["store"].unique())
        items = [item] if item is not None else sorted(df_feat["item"].unique())

        # Pre-extract entity histories and initial time index for recursive rollout
        entity_histories: Dict[tuple, list] = {}
        entity_time_idx: Dict[tuple, int] = {}

        for s in stores:
            for i in items:
                series_data = df_feat[(df_feat["store"] == s) & (df_feat["item"] == i)].sort_values("date")
                entity_histories[(s, i)] = list(series_data["sales"].values)
                entity_time_idx[(s, i)] = int(series_data["time_index"].iloc[-1])

        active_model = self.model.name if self.model is not None else "rolling_mean_baseline"
        all_forecast_records = []

        # Vectorized step-by-step recursive rollout across forecast horizon
        for step, d in enumerate(future_dates, 1):
            dow = d.day_of_week
            step_rows = []
            entity_keys = []

            for s in stores:
                for i in items:
                    hist = entity_histories[(s, i)]
                    t_idx = entity_time_idx[(s, i)] + step
                    row = {
                        "date": d,
                        "store": int(s),
                        "item": int(i),
                        "day": d.day,
                        "day_of_week": dow,
                        "week": d.isocalendar().week,
                        "month": d.month,
                        "quarter": d.quarter,
                        "year": d.year,
                        "is_weekend": int(dow >= 5),
                        "day_of_week_sin": float(np.sin(2 * np.pi * dow / 7.0)),
                        "day_of_week_cos": float(np.cos(2 * np.pi * dow / 7.0)),
                        "month_sin": float(np.sin(2 * np.pi * (d.month - 1) / 12.0)),
                        "month_cos": float(np.cos(2 * np.pi * (d.month - 1) / 12.0)),
                        "time_index": t_idx,
                        "sales_lag_1": float(hist[-1]),
                        "sales_lag_7": float(hist[-7] if len(hist) >= 7 else hist[-1]),
                        "sales_lag_14": float(hist[-14] if len(hist) >= 14 else hist[-1]),
                        "sales_lag_28": float(hist[-28] if len(hist) >= 28 else hist[-1]),
                        "sales_lag_56": float(hist[-56] if len(hist) >= 56 else hist[-1]),
                        "sales_lag_365": float(hist[-365] if len(hist) >= 365 else hist[-1]),
                        "rolling_mean_7": float(np.mean(hist[-7:])),
                        "rolling_mean_14": float(np.mean(hist[-14:])),
                        "rolling_mean_28": float(np.mean(hist[-28:])),
                        "rolling_mean_56": float(np.mean(hist[-56:])),
                        "rolling_std_7": float(np.std(hist[-7:])) if len(hist) >= 7 else 0.0,
                        "rolling_std_14": float(np.std(hist[-14:])) if len(hist) >= 14 else 0.0,
                        "rolling_std_28": float(np.std(hist[-28:])) if len(hist) >= 28 else 0.0,
                    }
                    step_rows.append(row)
                    entity_keys.append((s, i))

            df_step = pd.DataFrame(step_rows)
            # Ensure exact dtypes match trained feature matrix
            int_cols = ["store", "item", "day", "day_of_week", "week", "month", "quarter", "year", "is_weekend", "time_index"]
            for col in int_cols:
                df_step[col] = df_step[col].astype("int32")

            float_cols = [
                "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
                "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
                "sales_lag_56", "sales_lag_365", "rolling_mean_7", "rolling_mean_14",
                "rolling_mean_28", "rolling_mean_56", "rolling_std_7", "rolling_std_14", "rolling_std_28"
            ]
            for col in float_cols:
                df_step[col] = df_step[col].astype("float32")

            if self.model is not None and hasattr(self.model, "predict_features"):
                preds_step = self.model.predict_features(df_step[self.model.feature_names_])
            else:
                preds_step = df_step["rolling_mean_7"].values

            df_step["predicted_sales"] = preds_step

            # Conformal interval estimation per step
            if self.calibrator is not None:
                lower_step, upper_step = self.calibrator.predict_intervals(
                    preds_step, confidence_level=confidence_level
                )
            else:
                lower_step = np.maximum(0.0, preds_step * 0.85)
                upper_step = preds_step * 1.15

            df_step["lower_bound"] = lower_step
            df_step["upper_bound"] = upper_step

            # Append predictions back to entity histories for recursive multi-step forecasting
            for key, p_val in zip(entity_keys, preds_step, strict=False):
                entity_histories[key].append(float(p_val))

            all_forecast_records.append(df_step)

        inf_df = pd.concat(all_forecast_records, ignore_index=True)

        # Aggregate according to request granularity
        if store is not None and item is not None:
            agg_type = "individual_series"
            series_title = f"Store {store} — Item {item}"
            daily_forecast = inf_df.sort_values("date")
        elif store is not None and item is None:
            agg_type = "store_aggregated"
            series_title = f"Store {store} — Total Demand Across All Items"
            daily_forecast = inf_df.groupby("date").agg({
                "predicted_sales": "sum",
                "lower_bound": "sum",
                "upper_bound": "sum",
            }).reset_index()
        elif store is None and item is not None:
            agg_type = "item_aggregated"
            series_title = f"Item {item} — Total Demand Across All Stores"
            daily_forecast = inf_df.groupby("date").agg({
                "predicted_sales": "sum",
                "lower_bound": "sum",
                "upper_bound": "sum",
            }).reset_index()
        else:
            agg_type = "network_total"
            series_title = "Overall Network Demand Across All Stores & Items"
            daily_forecast = inf_df.groupby("date").agg({
                "predicted_sales": "sum",
                "lower_bound": "sum",
                "upper_bound": "sum",
            }).reset_index()

        daily_forecast["predicted_sales"] = np.round(daily_forecast["predicted_sales"], 2)
        daily_forecast["lower_bound"] = np.round(daily_forecast["lower_bound"], 2)
        daily_forecast["upper_bound"] = np.round(daily_forecast["upper_bound"], 2)

        return {
            "status": "success",
            "model": active_model,
            "aggregation_type": agg_type,
            "series_title": series_title,
            "store": store,
            "item": item,
            "horizon": horizon,
            "confidence_level": confidence_level,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dates": [str(d.date()) for d in daily_forecast["date"]],
            "predictions": daily_forecast["predicted_sales"].tolist(),
            "lower_bounds": daily_forecast["lower_bound"].tolist(),
            "upper_bounds": daily_forecast["upper_bound"].tolist(),
            "forecast_total": float(np.round(daily_forecast["predicted_sales"].sum(), 2)),
        }
