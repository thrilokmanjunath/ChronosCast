"""Exploratory Data Analysis (EDA) module for ChronosCast."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class TimeSeriesEDA:
    """Computes comprehensive analytical summaries across entities and time."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        if "date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"])

    def overall_summary(self) -> Dict[str, Any]:
        """Compute high-level descriptive metrics."""
        total_sales = float(self.df["sales"].sum())
        mean_daily_sales = float(self.df.groupby("date")["sales"].sum().mean())
        store_count = int(self.df["store"].nunique())
        item_count = int(self.df["item"].nunique())
        start_date = str(self.df["date"].min().date())
        end_date = str(self.df["date"].max().date())
        total_days = int((self.df["date"].max() - self.df["date"].min()).days + 1)

        return {
            "total_sales": total_sales,
            "mean_total_daily_sales": np.round(mean_daily_sales, 2),
            "store_count": store_count,
            "item_count": item_count,
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "total_observations": int(len(self.df)),
        }

    def store_level_summary(self) -> pd.DataFrame:
        """Calculate store-level aggregated sales, mean, standard deviation, and share."""
        store_stats = self.df.groupby("store")["sales"].agg(
            total_sales="sum",
            mean_sales="mean",
            std_sales="std",
            min_sales="min",
            max_sales="max",
        ).reset_index()
        total = store_stats["total_sales"].sum()
        store_stats["sales_share_pct"] = np.round((store_stats["total_sales"] / total) * 100, 2)
        store_stats["mean_sales"] = np.round(store_stats["mean_sales"], 2)
        store_stats["std_sales"] = np.round(store_stats["std_sales"], 2)
        return store_stats.sort_values(by="total_sales", ascending=False)

    def item_level_summary(self) -> pd.DataFrame:
        """Calculate product-level aggregated sales, velocity, and share."""
        item_stats = self.df.groupby("item")["sales"].agg(
            total_sales="sum",
            mean_sales="mean",
            std_sales="std",
            min_sales="min",
            max_sales="max",
        ).reset_index()
        total = item_stats["total_sales"].sum()
        item_stats["sales_share_pct"] = np.round((item_stats["total_sales"] / total) * 100, 2)
        item_stats["mean_sales"] = np.round(item_stats["mean_sales"], 2)
        item_stats["std_sales"] = np.round(item_stats["std_sales"], 2)
        return item_stats.sort_values(by="total_sales", ascending=False)

    def daily_aggregated_sales(self) -> pd.DataFrame:
        """Aggregate total sales across all stores and items per calendar day."""
        daily = self.df.groupby("date")["sales"].sum().reset_index()
        daily["rolling_7_mean"] = daily["sales"].rolling(7, min_periods=1).mean()
        daily["rolling_30_mean"] = daily["sales"].rolling(30, min_periods=1).mean()
        daily["rolling_90_mean"] = daily["sales"].rolling(90, min_periods=1).mean()
        daily["rolling_7_std"] = daily["sales"].rolling(7, min_periods=1).std().fillna(0)
        daily["rolling_30_std"] = daily["sales"].rolling(30, min_periods=1).std().fillna(0)
        return daily

    def seasonality_profiles(self) -> Dict[str, pd.DataFrame]:
        """Compute sales profiles across day-of-week, month, quarter, and year."""
        df_temp = self.df.copy()
        df_temp["day_of_week"] = df_temp["date"].dt.day_name()
        df_temp["day_of_week_num"] = df_temp["date"].dt.day_of_week
        df_temp["month_name"] = df_temp["date"].dt.month_name()
        df_temp["month"] = df_temp["date"].dt.month
        df_temp["quarter"] = df_temp["date"].dt.quarter
        df_temp["year"] = df_temp["date"].dt.year

        dow = df_temp.groupby(["day_of_week_num", "day_of_week"])["sales"].agg(
            mean_sales="mean", total_sales="sum"
        ).reset_index().sort_values("day_of_week_num")

        month = df_temp.groupby(["month", "month_name"])["sales"].agg(
            mean_sales="mean", total_sales="sum"
        ).reset_index().sort_values("month")

        quarter = df_temp.groupby("quarter")["sales"].agg(
            mean_sales="mean", total_sales="sum"
        ).reset_index().sort_values("quarter")

        year = df_temp.groupby("year")["sales"].agg(
            mean_sales="mean", total_sales="sum"
        ).reset_index().sort_values("year")
        year["yoy_growth_pct"] = np.round(year["total_sales"].pct_change() * 100, 2)

        # Monthly day-of-week heatmap matrix
        heatmap_matrix = df_temp.pivot_table(
            index="month_name",
            columns="day_of_week",
            values="sales",
            aggfunc="mean",
        )
        # Order columns Monday -> Sunday
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        heatmap_matrix = heatmap_matrix.reindex(index=month_order, columns=day_order)

        return {
            "day_of_week": dow,
            "month": month,
            "quarter": quarter,
            "year": year,
            "heatmap_matrix": heatmap_matrix,
        }
