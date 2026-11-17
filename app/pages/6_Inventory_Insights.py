"""ChronosCast — Inventory Risk & Replenishment Insights Page."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from app.utils import apply_custom_styles, render_sidebar_header
from chronoscast.forecasting.engine import ForecastingEngine

st.set_page_config(page_title="Inventory Insights — ChronosCast", page_icon="📦", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("📦 Prescriptive Inventory & Replenishment Insights")
st.markdown("Translate statistical demand projections and uncertainty intervals into actionable replenishment parameters.")

# Important Realistic Disclosure Banner
st.warning(
    "⚠️ **Business Modeling Disclaimer**: The underlying dataset contains historical point-of-sale transactions. "
    "It does not contain on-hand inventory levels, supplier lead times, unit costs, or stockout events. "
    "The metrics below are **hypothetical planning indicators** derived from forecast demand and uncertainty bounds."
)

@st.cache_resource
def get_forecasting_engine() -> ForecastingEngine:
    return ForecastingEngine()

engine = get_forecasting_engine()

# Planning parameters
with st.expander("⚙️ Replenishment & Supply Chain Assumptions", expanded=True):
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        store_id = st.selectbox("Store Location", list(range(1, 11)), index=0)
    with p2:
        item_id = st.selectbox("Product SKU", list(range(1, 51)), index=0)
    with p3:
        lead_time_days = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=21, value=7)
    with p4:
        service_level = st.selectbox("Target Service Level", [0.90, 0.95, 0.98, 0.99], index=1, format_func=lambda s: f"{int(s*100)}% Fill Rate")

# Generate 28-day forecast for the selected item
forecast_res = engine.generate_forecast(
    store=store_id,
    item=item_id,
    horizon=28,
    confidence_level=0.90,
)

preds = np.array(forecast_res["predictions"])
lower_bounds = np.array(forecast_res["lower_bounds"])
upper_bounds = np.array(forecast_res["upper_bounds"])

# Inventory Math Calculations
daily_mean_demand = float(np.mean(preds))
daily_demand_std = float(np.std(preds)) if np.std(preds) > 0 else float(daily_mean_demand * 0.25)
cv = daily_demand_std / max(1.0, daily_mean_demand)

# Lead time demand and safety stock
# mu_L = L * d_bar
lead_time_demand = lead_time_days * daily_mean_demand

# Z-score for service level
z_score = norm.ppf(service_level)
# Safety Stock: SS = Z * sigma * sqrt(L)
safety_stock = z_score * daily_demand_std * math.sqrt(lead_time_days)
# Reorder Point: ROP = mu_L + SS
reorder_point = lead_time_demand + safety_stock

# Stockout & Excess risk indicators
stockout_risk_score = min(100.0, (upper_bounds.max() / max(1.0, daily_mean_demand * 1.5)) * 50.0)
excess_risk_score = min(100.0, max(0.0, (1.0 - lower_bounds.min() / max(1.0, daily_mean_demand)) * 100.0))

# KPI Cards
st.subheader("Key Replenishment Recommendations")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Recommended Reorder Point (ROP)", f"{reorder_point:,.0f} Units", help="Trigger replenishment when net inventory reaches this level.")
with c2:
    st.metric("Buffer Safety Stock", f"{safety_stock:,.0f} Units", help=f"Buffer to guarantee {int(service_level*100)}% service level against volatility.")
with c3:
    st.metric("Lead Time Demand", f"{lead_time_demand:,.0f} Units", help=f"Expected consumption over {lead_time_days} days of transit.")
with c4:
    st.metric("Demand Volatility (CV)", f"{cv:.2f}", delta="Moderate" if cv < 0.3 else "High Volatility", delta_color="normal")

st.markdown("---")

# Visual: Replenishment Threshold Profile
st.subheader("Replenishment Threshold & Multi-Horizon Buffer")

dates = pd.to_datetime(forecast_res["dates"])
fig = go.Figure()

# Cumulative Expected Demand
cum_demand = np.cumsum(preds)
fig.add_trace(go.Scatter(
    x=dates, y=cum_demand,
    mode="lines+markers",
    name="Cumulative Projected Demand",
    line=dict(color="#2563EB", width=2.5),
))

# Upper cumulative bound
cum_upper = np.cumsum(upper_bounds)
fig.add_trace(go.Scatter(
    x=dates, y=cum_upper,
    mode="lines",
    line=dict(color="rgba(239, 68, 68, 0.4)", dash="dash"),
    name="90% Cumulative Surge (Peak Exposure)",
))

# Lead Time Marker Line
lead_date = dates[min(lead_time_days - 1, len(dates) - 1)]
fig.add_vline(x=lead_date, line_dash="dot", line_color="#F59E0B", annotation_text=f"Lead Time Arrival ({lead_time_days}D)")

fig.update_layout(
    template="plotly_white",
    height=400,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Cumulative Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Risk Matrix Columns
r1, r2 = st.columns(2)
with r1:
    st.markdown("#### 🚨 Stockout Exposure Analysis")
    st.write(f"Based on the **90% upper conformal quantile**, peak daily surge can reach **{upper_bounds.max():.1f} units** (compared to the daily mean of **{daily_mean_demand:.1f} units**).")
    st.progress(min(1.0, stockout_risk_score / 100.0))
    st.caption(f"Calculated Surge Risk Index: **{stockout_risk_score:.1f} / 100**")

with r2:
    st.markdown("#### 📦 Working Capital & Excess Inventory Risk")
    st.write(f"Trough demand in the horizon drops to **{lower_bounds.min():.1f} units**. Holding excessive cycle stock during mid-week troughs inflates holding costs.")
    st.progress(min(1.0, excess_risk_score / 100.0))
    st.caption(f"Calculated Idle Capital Index: **{excess_risk_score:.1f} / 100**")
