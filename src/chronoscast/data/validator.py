"""Data validation module for ChronosCast."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataQualityValidator:
    """Validates schema, completeness, integrity, and temporal consistency."""

    def __init__(
        self,
        expected_stores: int = 10,
        expected_items: int = 50,
        expected_start_date: str = "2013-01-01",
        expected_end_date: str = "2017-12-31",
    ) -> None:
        self.expected_stores = expected_stores
        self.expected_items = expected_items
        self.expected_start_date = pd.to_datetime(expected_start_date)
        self.expected_end_date = pd.to_datetime(expected_end_date)
        self.expected_days = (self.expected_end_date - self.expected_start_date).days + 1
        self.expected_total_rows = self.expected_stores * self.expected_items * self.expected_days

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks on the dataframe."""
        report: Dict[str, Any] = {
            "total_rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "checks": {},
            "passed": True,
        }

        # 1. Null check
        null_counts = df.isnull().sum().to_dict()
        report["null_counts"] = {k: int(v) for k, v in null_counts.items()}
        has_nulls = sum(null_counts.values()) > 0
        report["checks"]["no_missing_values"] = not has_nulls
        if has_nulls:
            report["passed"] = False

        # 2. Duplicate check (store, item, date)
        dup_count = int(df.duplicated(subset=["date", "store", "item"]).sum())
        report["duplicate_keys"] = dup_count
        report["checks"]["no_duplicate_keys"] = dup_count == 0
        if dup_count > 0:
            report["passed"] = False

        # 3. Date validity & coverage
        dates = pd.to_datetime(df["date"])
        min_date = dates.min()
        max_date = dates.max()
        report["date_range"] = {
            "min_date": str(min_date.date()),
            "max_date": str(max_date.date()),
            "unique_days": int(dates.nunique()),
            "expected_days": int(self.expected_days),
        }
        report["checks"]["date_range_valid"] = (
            min_date == self.expected_start_date and max_date == self.expected_end_date
        )
        report["checks"]["all_calendar_days_present"] = dates.nunique() == self.expected_days
        if not (report["checks"]["date_range_valid"] and report["checks"]["all_calendar_days_present"]):
            report["passed"] = False

        # 4. Stores and items identifiers
        unique_stores = sorted(df["store"].unique().tolist())
        unique_items = sorted(df["item"].unique().tolist())
        report["entity_counts"] = {
            "unique_stores": int(len(unique_stores)),
            "unique_items": int(len(unique_items)),
            "total_series": int(len(unique_stores) * len(unique_items)),
        }
        report["checks"]["store_identifiers_valid"] = (
            len(unique_stores) == self.expected_stores
            and unique_stores == list(range(1, self.expected_stores + 1))
        )
        report["checks"]["item_identifiers_valid"] = (
            len(unique_items) == self.expected_items
            and unique_items == list(range(1, self.expected_items + 1))
        )
        if not (report["checks"]["store_identifiers_valid"] and report["checks"]["item_identifiers_valid"]):
            report["passed"] = False

        # 5. Missing combinations check
        expected_combinations = self.expected_stores * self.expected_items * self.expected_days
        report["expected_total_rows"] = int(expected_combinations)
        report["checks"]["full_temporal_grid_intact"] = len(df) == expected_combinations
        if not report["checks"]["full_temporal_grid_intact"]:
            report["passed"] = False

        # 6. Sales target validity
        sales = df["sales"]
        min_sales = float(sales.min())
        max_sales = float(sales.max())
        neg_sales = int((sales < 0).sum())
        nan_sales = int(sales.isna().sum())
        report["sales_distribution"] = {
            "min": min_sales,
            "max": max_sales,
            "mean": float(np.round(sales.mean(), 2)),
            "std": float(np.round(sales.std(), 2)),
            "median": float(sales.median()),
            "negative_count": neg_sales,
            "nan_count": nan_sales,
        }
        report["checks"]["non_negative_sales"] = neg_sales == 0
        report["checks"]["no_nan_sales"] = nan_sales == 0
        if neg_sales > 0 or nan_sales > 0:
            report["passed"] = False

        logger.info(
            "Validation finished. Overall passed: %s. Total rows: %d.",
            report["passed"],
            report["total_rows"],
        )
        return report
