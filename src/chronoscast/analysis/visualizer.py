"""Plotly & Matplotlib visualization generator for ChronosCast."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chronoscast.analysis.diagnostics import compute_acf_pacf, perform_decomposition
from chronoscast.analysis.eda import TimeSeriesEDA
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class TimeSeriesVisualizer:
    """Creates interactive Plotly figures and exports publication-ready static/HTML charts."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.eda = TimeSeriesEDA(df)

    def plot_overall_demand_trend(self) -> go.Figure:
        """Plot total historical daily sales with 7, 30, and 90-day rolling averages."""
        daily = self.eda.daily_aggregated_sales()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["sales"],
                name="Daily Actual Sales",
                mode="lines",
                line=dict(color="rgba(100, 116, 139, 0.4)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["rolling_7_mean"],
                name="7-Day Rolling Mean",
                mode="lines",
                line=dict(color="#2563EB", width=1.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["rolling_30_mean"],
                name="30-Day Rolling Mean",
                mode="lines",
                line=dict(color="#10B981", width=2.2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["rolling_90_mean"],
                name="90-Day Rolling Mean",
                mode="lines",
                line=dict(color="#F59E0B", width=2.5),
            )
        )

        fig.update_layout(
            title="ChronosCast — Total Daily Retail Demand & Long-Term Trend (2013–2017)",
            xaxis_title="Calendar Date",
            yaxis_title="Total Units Sold Across All Stores & Items",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def plot_store_comparison(self) -> go.Figure:
        """Plot total sales volume and average demand across all 10 stores."""
        store_stats = self.eda.store_level_summary()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total Demand by Store (Units)", "Average Daily Store Demand (Units)"),
        )
        fig.add_trace(
            go.Bar(
                x=store_stats["store"].astype(str),
                y=store_stats["total_sales"],
                name="Total Sales",
                marker_color="#3B82F6",
                text=store_stats["sales_share_pct"].apply(lambda v: f"{v}%"),
                textposition="auto",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=store_stats["store"].astype(str),
                y=store_stats["mean_sales"],
                name="Mean Sales",
                marker_color="#10B981",
                text=store_stats["mean_sales"].astype(str),
                textposition="auto",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            title_text="Store Demand Heterogeneity & Volume Comparison",
            template="plotly_white",
            showlegend=False,
        )
        fig.update_xaxes(title_text="Store ID", row=1, col=1)
        fig.update_xaxes(title_text="Store ID", row=1, col=2)
        return fig

    def plot_product_comparison(self, top_n: int = 10) -> go.Figure:
        """Plot top N vs bottom N selling items."""
        item_stats = self.eda.item_level_summary()
        top_items = item_stats.head(top_n)
        bottom_items = item_stats.tail(top_n).iloc[::-1]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"Top {top_n} Highest Volume Items", f"Bottom {top_n} Slower Moving Items"),
        )
        fig.add_trace(
            go.Bar(
                y=top_items["item"].astype(str),
                x=top_items["total_sales"],
                orientation="h",
                name="High Volume",
                marker_color="#4F46E5",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                y=bottom_items["item"].astype(str),
                x=bottom_items["total_sales"],
                orientation="h",
                name="Slow Volume",
                marker_color="#EC4899",
            ),
            row=1, col=2,
        )
        fig.update_layout(
            title_text="Product Demand Velocity Distribution",
            template="plotly_white",
            showlegend=False,
        )
        fig.update_yaxes(type="category", title_text="Item ID", row=1, col=1)
        fig.update_yaxes(type="category", title_text="Item ID", row=1, col=2)
        return fig

    def plot_seasonal_heatmap(self) -> go.Figure:
        """Create heatmap showing average sales across months and day-of-week."""
        profiles = self.eda.seasonality_profiles()
        matrix = profiles["heatmap_matrix"]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix.values,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="Viridis",
                colorbar=dict(title="Average Sales"),
            )
        )
        fig.update_layout(
            title="Seasonal Interaction: Average Daily Sales by Month & Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Month",
            template="plotly_white",
        )
        return fig

    def plot_decomposition(self, series: pd.Series, period: int = 7) -> go.Figure:
        """Plot classical decomposition components: Observed, Trend, Seasonal, Residual."""
        decomp = perform_decomposition(series, model="additive", period=period)

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Observed Demand", "Trend Component", "Seasonal Component (Weekly)", "Residual Component"),
            vertical_spacing=0.06,
        )
        fig.add_trace(
            go.Scatter(x=decomp["observed"].index, y=decomp["observed"], line=dict(color="#1F2937", width=1)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=decomp["trend"].index, y=decomp["trend"], line=dict(color="#2563EB", width=2)),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=decomp["seasonal"].index, y=decomp["seasonal"], line=dict(color="#10B981", width=1.5)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=decomp["resid"].index, y=decomp["resid"], line=dict(color="#EF4444", width=1)),
            row=4, col=1,
        )
        fig.update_layout(
            title=f"Additive Seasonal Decomposition (Period = {period} Days)",
            template="plotly_white",
            showlegend=False,
            height=850,
        )
        return fig

    def plot_acf_pacf(self, series: pd.Series, nlags: int = 35) -> go.Figure:
        """Plot ACF and PACF with 95% Bartlett confidence bands."""
        res = compute_acf_pacf(series, nlags=nlags)
        lags = res["lags"]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Autocorrelation Function (ACF)", "Partial Autocorrelation Function (PACF)"),
        )
        # ACF
        fig.add_trace(
            go.Bar(x=lags, y=res["acf"], name="ACF", marker_color="#3B82F6", width=0.3),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=lags, y=res["acf_confint_upper"],
                mode="lines", line=dict(color="rgba(239, 68, 68, 0.7)", dash="dash"),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=lags, y=res["acf_confint_lower"],
                mode="lines", line=dict(color="rgba(239, 68, 68, 0.7)", dash="dash"),
                showlegend=False,
            ),
            row=1, col=1,
        )

        # PACF
        fig.add_trace(
            go.Bar(x=lags, y=res["pacf"], name="PACF", marker_color="#10B981", width=0.3),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lags, y=res["pacf_confint_upper"],
                mode="lines", line=dict(color="rgba(239, 68, 68, 0.7)", dash="dash"),
                showlegend=False,
            ),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lags, y=res["pacf_confint_lower"],
                mode="lines", line=dict(color="rgba(239, 68, 68, 0.7)", dash="dash"),
                showlegend=False,
            ),
            row=1, col=2,
        )

        fig.update_layout(
            title=f"Sample Autocorrelation Diagnostics (Up to {nlags} Daily Lags)",
            template="plotly_white",
            showlegend=False,
        )
        fig.update_xaxes(title_text="Lag (Days)", row=1, col=1)
        fig.update_xaxes(title_text="Lag (Days)", row=1, col=2)
        fig.update_yaxes(title_text="Correlation", row=1, col=1)
        fig.update_yaxes(title_text="Partial Correlation", row=1, col=2)
        return fig
