"""Generate all exploratory analysis artifacts, diagnostic tests, and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from chronoscast.analysis.diagnostics import (
    perform_decomposition,
    run_stationarity_tests,
)
from chronoscast.analysis.eda import TimeSeriesEDA
from chronoscast.analysis.visualizer import TimeSeriesVisualizer
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_eda_artifacts(
    data_path: str = "data/interim/cleaned_sales.parquet",
    figures_dir: str = "reports/figures",
    reports_dir: str = "reports/data_quality",
) -> None:
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned dataset from %s...", data_path)
    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    eda = TimeSeriesEDA(df)
    viz = TimeSeriesVisualizer(df)

    # 1. Summaries
    logger.info("Computing EDA summaries...")
    overall = eda.overall_summary()
    store_stats = eda.store_level_summary()
    item_stats = eda.item_level_summary()
    daily = eda.daily_aggregated_sales()

    with open(rep_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    store_stats.to_csv(rep_dir / "store_summary.csv", index=False)
    item_stats.to_csv(rep_dir / "item_summary.csv", index=False)

    # 2. Stationarity Diagnostics on Overall Series and Sample Series (Store 1, Item 1)
    logger.info("Running Stationarity diagnostics (ADF & KPSS)...")
    daily_indexed = daily.set_index("date")["sales"]
    stationarity_overall = run_stationarity_tests(daily_indexed)

    # Sample single series (Store 1, Item 1)
    s1i1 = df[(df["store"] == 1) & (df["item"] == 1)].sort_values("date").set_index("date")["sales"]
    stationarity_s1i1 = run_stationarity_tests(s1i1)

    # Differenced series (1-day diff)
    s1i1_diff = s1i1.diff().dropna()
    stationarity_s1i1_diff = run_stationarity_tests(s1i1_diff)

    stationarity_report = {
        "overall_aggregate_demand": stationarity_overall,
        "sample_series_store1_item1": stationarity_s1i1,
        "sample_series_store1_item1_diff_1": stationarity_s1i1_diff,
    }
    with open(rep_dir / "stationarity_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(stationarity_report, f, indent=2)

    logger.info("Overall stationarity: %s", stationarity_overall["conclusion"])
    logger.info("Store 1 Item 1 stationarity: %s", stationarity_s1i1["conclusion"])

    # 3. Interactive Plotly Figures
    logger.info("Generating Plotly interactive figures...")
    fig_trend = viz.plot_overall_demand_trend()
    fig_trend.write_html(str(fig_dir / "overall_demand_trend.html"))

    fig_store = viz.plot_store_comparison()
    fig_store.write_html(str(fig_dir / "store_comparison.html"))

    fig_product = viz.plot_product_comparison()
    fig_product.write_html(str(fig_dir / "product_comparison.html"))

    fig_heatmap = viz.plot_seasonal_heatmap()
    fig_heatmap.write_html(str(fig_dir / "seasonal_heatmap.html"))

    fig_decomp = viz.plot_decomposition(daily_indexed)
    fig_decomp.write_html(str(fig_dir / "seasonal_decomposition.html"))

    fig_acf = viz.plot_acf_pacf(s1i1)
    fig_acf.write_html(str(fig_dir / "acf_pacf.html"))

    # 4. Generate Matplotlib static PNGs for reliable embedding
    logger.info("Generating Matplotlib high-resolution static PNG figures...")

    # Overall Demand PNG
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(daily["date"], daily["sales"], color="#94A3B8", alpha=0.5, label="Daily Total Sales", linewidth=0.8)
    plt.plot(daily["date"], daily["rolling_7_mean"], color="#2563EB", label="7-Day Rolling Mean", linewidth=1.5)
    plt.plot(daily["date"], daily["rolling_30_mean"], color="#10B981", label="30-Day Rolling Mean", linewidth=2.0)
    plt.plot(daily["date"], daily["rolling_90_mean"], color="#F59E0B", label="90-Day Rolling Mean", linewidth=2.2)
    plt.title("ChronosCast — Historical Daily Demand Trend (2013-2017)", fontsize=13, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Total Demand Across All 10 Stores & 50 Items")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "overall_demand_trend.png")
    plt.close()

    # Store Volume PNG
    plt.figure(figsize=(10, 4.5), dpi=150)
    plt.bar(store_stats["store"].astype(str), store_stats["total_sales"] / 1e6, color="#3B82F6", edgecolor="#1D4ED8")
    plt.title("Total Sales Volume by Store (Millions of Units)", fontsize=12, fontweight="bold")
    plt.xlabel("Store ID")
    plt.ylabel("Total Sales (M Units)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "store_comparison.png")
    plt.close()

    # Heatmap PNG
    profiles = eda.seasonality_profiles()
    matrix = profiles["heatmap_matrix"]
    plt.figure(figsize=(9, 6), dpi=150)
    im = plt.imshow(matrix.values, cmap="viridis", aspect="auto")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar(im, label="Mean Sales per Day")
    plt.title("Average Daily Demand: Month vs Day of Week", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "seasonal_heatmap.png")
    plt.close()

    # Decomposition PNG
    decomp = perform_decomposition(daily_indexed, period=7)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, dpi=150)
    axes[0].plot(decomp["observed"].index, decomp["observed"], color="#1E293B", lw=1)
    axes[0].set_ylabel("Observed")
    axes[1].plot(decomp["trend"].index, decomp["trend"], color="#2563EB", lw=1.8)
    axes[1].set_ylabel("Trend")
    axes[2].plot(decomp["seasonal"].index, decomp["seasonal"], color="#10B981", lw=1.2)
    axes[2].set_ylabel("Seasonal (7D)")
    axes[3].plot(decomp["resid"].index, decomp["resid"], color="#EF4444", lw=0.8)
    axes[3].set_ylabel("Residual")
    fig.suptitle("Additive Time Series Decomposition (Total Demand)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "seasonal_decomposition.png")
    plt.close()

    logger.info("All EDA artifacts and figures generated successfully in %s.", fig_dir)


if __name__ == "__main__":
    generate_eda_artifacts()
