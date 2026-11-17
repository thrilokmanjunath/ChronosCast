"""ChronosCast — Data Explorer Page."""

import plotly.graph_objects as go
import streamlit as st

from app.utils import apply_custom_styles, load_interim_data, render_sidebar_header

st.set_page_config(page_title="Data Explorer — ChronosCast", page_icon="🔍", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("🔍 Historical Data Explorer")
st.markdown("Filter, visualize, and inspect historical sales observations across 10 stores and 50 products.")

df = load_interim_data()

# Filters in sidebar / top container
with st.container():
    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        stores = ["All Stores"] + sorted(df["store"].unique().tolist())
        selected_store = st.selectbox("Store Filter", stores, index=0)
    with fcol2:
        items = ["All Items"] + sorted(df["item"].unique().tolist())
        selected_item = st.selectbox("Product Filter", items, index=0)
    with fcol3:
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Apply filters
filtered_df = df.copy()
if selected_store != "All Stores":
    filtered_df = filtered_df[filtered_df["store"] == selected_store]
if selected_item != "All Items":
    filtered_df = filtered_df[filtered_df["item"] == selected_item]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = filtered_df[(filtered_df["date"].dt.date >= start_d) & (filtered_df["date"].dt.date <= end_d)]

# Aggregation for plotting
daily_filtered = filtered_df.groupby("date")["sales"].sum().reset_index()
daily_filtered["rolling_7"] = daily_filtered["sales"].rolling(7, min_periods=1).mean()
daily_filtered["rolling_30"] = daily_filtered["sales"].rolling(30, min_periods=1).mean()

# Filtered KPIs
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Filtered Volume", f"{daily_filtered['sales'].sum():,.0f} Units")
with k2:
    st.metric("Daily Mean Sales", f"{daily_filtered['sales'].mean():.1f} Units")
with k3:
    st.metric("Daily Volatility (Std)", f"{daily_filtered['sales'].std():.1f} Units")
with k4:
    st.metric("Active Series Count", f"{filtered_df.groupby(['store', 'item']).ngroups} Series")

# Interactive Plot
st.subheader("Sales Demand Trajectory")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_filtered["date"], y=daily_filtered["sales"],
    mode="lines", name="Actual Daily Units",
    line=dict(color="#94A3B8", width=1.2),
))
fig.add_trace(go.Scatter(
    x=daily_filtered["date"], y=daily_filtered["rolling_7"],
    mode="lines", name="7-Day Rolling Mean",
    line=dict(color="#2563EB", width=2),
))
fig.add_trace(go.Scatter(
    x=daily_filtered["date"], y=daily_filtered["rolling_30"],
    mode="lines", name="30-Day Rolling Mean",
    line=dict(color="#10B981", width=2.2),
))
fig.update_layout(
    template="plotly_white",
    height=420,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Date",
    yaxis_title="Units Sold",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# Data Table & Export
st.subheader("Filtered Observations Table")
st.dataframe(filtered_df.sort_values(by=["date", "store", "item"]).head(1000), use_container_width=True)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Export Filtered Observations (CSV)",
    data=csv_data,
    file_name=f"chronoscast_filtered_{selected_store}_{selected_item}.csv",
    mime="text/csv",
)
