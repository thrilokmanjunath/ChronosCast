"""ChronosCast — Forecast Studio Page."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.utils import apply_custom_styles, load_interim_data, render_sidebar_header
from chronoscast.forecasting.engine import ForecastingEngine

st.set_page_config(page_title="Forecast Studio — ChronosCast", page_icon="🔮", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("🔮 Interactive Forecast Studio")
st.markdown("Generate out-of-sample forward projections and conformal prediction intervals across multiple granularities.")

@st.cache_resource
def get_forecasting_engine() -> ForecastingEngine:
    return ForecastingEngine()

engine = get_forecasting_engine()
df_hist = load_interim_data()

# Control Panel
with st.expander("🛠️ Forecast Configuration Parameters", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        store_opt = st.selectbox("Store Entity", ["All Stores (Network Aggregate)"] + list(range(1, 11)), index=1)
        store_val = None if store_opt == "All Stores (Network Aggregate)" else int(store_opt)
    with c2:
        item_opt = st.selectbox("Product SKU", ["All Products (Aggregate)"] + list(range(1, 51)), index=1)
        item_val = None if item_opt == "All Products (Aggregate)" else int(item_opt)
    with c3:
        horizon = st.select_slider("Forecast Horizon", options=[7, 14, 28], value=14, format_func=lambda h: f"{h} Days")
    with c4:
        confidence = st.select_slider("Uncertainty Confidence Level", options=[0.80, 0.90, 0.95], value=0.90, format_func=lambda c: f"{int(c*100)}% Band")

    run_btn = st.button("🚀 Generate Forecast & Prediction Intervals", use_container_width=True)

if run_btn or "last_forecast" not in st.session_state:
    with st.spinner("Executing dynamic model inference and uncertainty calibration..."):
        forecast_res = engine.generate_forecast(
            store=store_val,
            item=item_val,
            horizon=horizon,
            confidence_level=confidence,
        )
        st.session_state["last_forecast"] = forecast_res

res = st.session_state["last_forecast"]

# Summary Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Projected Volume", f"{res['forecast_total']:,.0f} Units")
with m2:
    mean_daily = res['forecast_total'] / res['horizon']
    st.metric("Projected Daily Average", f"{mean_daily:,.1f} Units/Day")
with m3:
    half_width = (res['upper_bounds'][0] - res['lower_bounds'][0]) / 2.0
    st.metric(f"{int(res['confidence_level']*100)}% Margin of Error", f"±{half_width:,.1f} Units")
with m4:
    st.metric("Active Model", res['model'].upper())

st.markdown("---")

# Visualizing Forecast with Historical Context
st.subheader(f"📈 Forward Projection: {res['series_title']}")

# Extract historical context for the same entity granularity (last 60 days)
last_hist_date = df_hist["date"].max()
hist_start = last_hist_date - pd.Timedelta(days=60)
sub_df = df_hist[df_hist["date"] >= hist_start].copy()
if store_val is not None:
    sub_df = sub_df[sub_df["store"] == store_val]
if item_val is not None:
    sub_df = sub_df[sub_df["item"] == item_val]

hist_daily = sub_df.groupby("date")["sales"].sum().reset_index()

# Plotly combined visualization
fig = go.Figure()

# Historical Line
fig.add_trace(go.Scatter(
    x=hist_daily["date"], y=hist_daily["sales"],
    name="Historical Actual Sales",
    mode="lines+markers",
    line=dict(color="#64748B", width=1.5),
    marker=dict(size=4),
))

# Forecast Dates
forecast_dates = pd.to_datetime(res["dates"])

# Uncertainty Band (Upper and Lower bounds)
fig.add_trace(go.Scatter(
    x=forecast_dates, y=res["upper_bounds"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    name="Upper Bound",
))
fig.add_trace(go.Scatter(
    x=forecast_dates, y=res["lower_bounds"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(37, 99, 235, 0.2)",
    name=f"{int(res['confidence_level']*100)}% Conformal Prediction Band",
))

# Point Forecast Line
fig.add_trace(go.Scatter(
    x=forecast_dates, y=res["predictions"],
    name="Point Forecast",
    mode="lines+markers",
    line=dict(color="#2563EB", width=2.5),
    marker=dict(size=6, symbol="diamond"),
))

fig.update_layout(
    template="plotly_white",
    height=480,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Demand Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Forecast Table
st.subheader("Forecast Values & Interval Table")
forecast_table = pd.DataFrame({
    "Date": res["dates"],
    "Day of Week": pd.to_datetime(res["dates"]).day_name(),
    "Point Forecast": res["predictions"],
    f"Lower Bound ({int(res['confidence_level']*100)}%)": res["lower_bounds"],
    f"Upper Bound ({int(res['confidence_level']*100)}%)": res["upper_bounds"],
})
st.dataframe(forecast_table, use_container_width=True)

csv_forecast = forecast_table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Forecast CSV",
    data=csv_forecast,
    file_name=f"chronoscast_forecast_s{store_val}_i{item_val}_h{horizon}.csv",
    mime="text/csv",
)
