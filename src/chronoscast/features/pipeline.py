"""Feature engineering pipeline for ChronosCast.

Generates leakage-free lag, rolling, calendar, entity, and trend features
independently within each (store, item) series.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_LAGS = [1, 7, 14, 28, 56, 365]
DEFAULT_ROLLING_MEANS = [7, 14, 28, 56]
DEFAULT_ROLLING_STDS = [7, 14, 28]


class TimeSeriesFeaturePipeline:
    """Production-grade feature engineering pipeline for store-item demand forecasting.

    Guarantees:
    1. Strict grouping by (store, item).
    2. Observations shifted before rolling window calculations to eliminate target leakage.
    3. Calendar and cyclic temporal features.
    4. Monotonic time_index for trend capture.
    5. Identical feature structure across training and inference.
    """

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_means: Optional[List[int]] = None,
        rolling_stds: Optional[List[int]] = None,
        target_col: str = "sales",
        date_col: str = "date",
        store_col: str = "store",
        item_col: str = "item",
    ) -> None:
        self.lags = lags if lags is not None else DEFAULT_LAGS
        self.rolling_means = rolling_means if rolling_means is not None else DEFAULT_ROLLING_MEANS
        self.rolling_stds = rolling_stds if rolling_stds is not None else DEFAULT_ROLLING_STDS
        self.target_col = target_col
        self.date_col = date_col
        self.store_col = store_col
        self.item_col = item_col
        self.reference_date: Optional[pd.Timestamp] = None
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> TimeSeriesFeaturePipeline:
        """Fit pipeline parameters (reference start date)."""
        dates = pd.to_datetime(df[self.date_col])
        self.reference_date = dates.min()
        logger.info("Fitted feature pipeline with reference date: %s", self.reference_date.date())
        return self

    def _add_calendar_and_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar and deterministic time trend features."""
        df = df.copy()
        dates = pd.to_datetime(df[self.date_col])

        # Calendar features
        df["day"] = dates.dt.day.astype("int32")
        df["day_of_week"] = dates.dt.day_of_week.astype("int32")
        # isocalendar week handles leap years
        df["week"] = dates.dt.isocalendar().week.astype("int32")
        df["month"] = dates.dt.month.astype("int32")
        df["quarter"] = dates.dt.quarter.astype("int32")
        df["year"] = dates.dt.year.astype("int32")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int32")

        # Cyclical transforms for day_of_week and month
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0).astype("float32")

        # Monotonic time index from reference date
        if self.reference_date is not None:
            df["time_index"] = (dates - self.reference_date).dt.days.astype("int32")
        else:
            df["time_index"] = (dates - dates.min()).dt.days.astype("int32")

        return df

    def transform(self, df: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
        """Transform dataframe by engineering lag and rolling features per entity.

        Args:
            df: Input DataFrame with date, store, item, sales.
            drop_na: Whether to drop rows with NaN values resulting from lags.

        Returns:
            DataFrame with enriched features.
        """
        logger.info("Transforming dataframe (%d rows) with feature pipeline...", len(df))

        # Sort strictly by entity and date
        df = df.sort_values(by=[self.store_col, self.item_col, self.date_col]).copy()

        # Add calendar and trend features
        df = self._add_calendar_and_trend(df)

        # Entity categoricals
        df[self.store_col] = df[self.store_col].astype("category")
        df[self.item_col] = df[self.item_col].astype("category")

        # Groupby object for entity-level operations
        grouped = df.groupby([self.store_col, self.item_col], observed=True)[self.target_col]

        # 1. Lag features: sales_lag_k
        for lag in self.lags:
            col_name = f"sales_lag_{lag}"
            df[col_name] = grouped.shift(lag).astype("float32")

        # 2. Rolling features: MUST shift by 1 first to prevent current-day target leakage!
        # shift(1) ensures rolling_mean_7 at time t uses sales at [t-7, ..., t-1]
        shifted_target = grouped.shift(1)
        grouped_shifted = shifted_target.groupby([df[self.store_col], df[self.item_col]], observed=True)

        for w in self.rolling_means:
            col_name = f"rolling_mean_{w}"
            df[col_name] = grouped_shifted.transform(
                lambda x, window=w: x.rolling(window, min_periods=1).mean()
            ).astype("float32")

        for w in self.rolling_stds:
            col_name = f"rolling_std_{w}"
            df[col_name] = (
                grouped_shifted.transform(lambda x, window=w: x.rolling(window, min_periods=2).std())
                .fillna(0.0)
                .astype("float32")
            )

        # Record feature column names (excluding identifiers and target)
        exclude_cols = {self.date_col, self.target_col}
        self.feature_names_ = [c for c in df.columns if c not in exclude_cols]

        if drop_na:
            initial_len = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info("Dropped %d NaN rows from warm-up period. Remaining: %d rows.", initial_len - len(df), len(df))
        else:
            df = df.reset_index(drop=True)

        return df

    def fit_transform(self, df: pd.DataFrame, drop_na: bool = False) -> pd.DataFrame:
        """Fit pipeline and transform in one step."""
        return self.fit(df).transform(df, drop_na=drop_na)

    def get_feature_names(self) -> List[str]:
        """Return list of generated feature column names."""
        return list(self.feature_names_)
