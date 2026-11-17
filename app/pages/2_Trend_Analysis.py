"""ChronosCast — Trend & Seasonality Analysis Page."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import apply_custom_styles, load_interim_data, render_sidebar_header
from chronoscast.analysis.diagnostics import compute_acf_pacf, perform_decomposition
from chronoscast.analysis.eda import TimeSeriesEDA

st.set_page_config(page_title="Trend & Seasonality — ChronosCast", page_icon="📊", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("📊 Trend, Seasonality & Diagnostics")
st.markdown("Deep-dive decomposition, periodic seasonality patterns, stationarity tests, and autocorrelation dynamics.")

df = load_interim_data()
eda = TimeSeriesEDA(df)

tab1, tab2, tab3, tab4 = st.tabs(["Periodic Seasonality", "Time Series Decomposition", "Store & SKU Dynamics", "Stationarity & ACF"])

# ----------------- Tab 1: Seasonality -----------------
with tab1:
    st.subheader("Seasonal Heatmap: Month vs Day of Week")
    profiles = eda.seasonality_profiles()
    matrix = profiles["heatmap_matrix"]

    fig_heat = px.imshow(
        matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        color_continuous_scale="Viridis",
        labels=dict(x="Day of Week", y="Month", color="Mean Sales"),
        aspect="auto",
    )
    fig_heat.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    col_dow, col_month = st.columns(2)
    with col_dow:
        st.subheader("Day-of-Week Effect")
        dow_df = profiles["day_of_week"]
        fig_dow = px.bar(
            dow_df, x="day_of_week", y="mean_sales",
            color="mean_sales", color_continuous_scale="Blues",
            labels={"day_of_week": "Day", "mean_sales": "Average Units Sold"},
        )
        fig_dow.update_layout(template="plotly_white", showlegend=False, height=350)
        st.plotly_chart(fig_dow, use_container_width=True)

    with col_month:
        st.subheader("Monthly Demand Profile")
        m_df = profiles["month"]
        fig_m = px.bar(
            m_df, x="month_name", y="mean_sales",
            color="mean_sales", color_continuous_scale="Greens",
            labels={"month_name": "Month", "mean_sales": "Average Units Sold"},
        )
        fig_m.update_layout(template="plotly_white", showlegend=False, height=350)
        st.plotly_chart(fig_m, use_container_width=True)

# ----------------- Tab 2: Decomposition -----------------
with tab2:
    st.subheader("Classical Additive Decomposition (Total Demand)")
    daily = eda.daily_aggregated_sales().set_index("date")["sales"]
    decomp = perform_decomposition(daily, model="additive", period=7)

    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Scatter(x=decomp["observed"].index, y=decomp["observed"], name="Observed", line=dict(color="#1E293B", width=1)))
        fig_decomp.add_trace(go.Scatter(x=decomp["trend"].index, y=decomp["trend"], name="Trend Component", line=dict(color="#2563EB", width=2)))
        fig_decomp.add_trace(go.Scatter(x=decomp["seasonal"].index, y=decomp["seasonal"], name="Seasonal (7-Day)", line=dict(color="#10B981", width=1.5)))
        fig_decomp.add_trace(go.Scatter(x=decomp["resid"].index, y=decomp["resid"], name="Residuals", line=dict(color="#EF4444", width=1)))
        fig_decomp.update_layout(
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

    with d_col2:
        st.markdown("#### Component Diagnostics")
        st.metric("Observed Mean", f"{decomp['observed'].mean():,.1f}")
        st.metric("Weekly Seasonal Range", f"{decomp['seasonal'].max() - decomp['seasonal'].min():,.1f}")
        st.metric("Residual Std Dev", f"{decomp['resid'].std():,.1f}")
        st.caption("Weekly additive cycle accounts for regular weekend surges and weekday troughs.")

# ----------------- Tab 3: Store & SKU Dynamics -----------------
with tab3:
    st.subheader("Store & Product Volume Dynamics")
    col_store, col_item = st.columns(2)

    with col_store:
        st.markdown("#### Store Volume Distribution")
        store_stats = eda.store_level_summary()
        fig_s = px.bar(
            store_stats, x="store", y="total_sales",
            text="sales_share_pct",
            color="total_sales", color_continuous_scale="Blues",
            labels={"store": "Store ID", "total_sales": "Total Sales (Units)"},
        )
        fig_s.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_s.update_layout(template="plotly_white", showlegend=False, height=380)
        st.plotly_chart(fig_s, use_container_width=True)

    with col_item:
        st.markdown("#### Top 10 vs Bottom 10 Products")
        item_stats = eda.item_level_summary()
        top_items = item_stats.head(10).copy()
        top_items["tier"] = "Top 10"
        bottom_items = item_stats.tail(10).copy()
        bottom_items["tier"] = "Bottom 10"
        combo = pd.concat([top_items, bottom_items])
        fig_i = px.bar(
            combo, x="total_sales", y=combo["item"].astype(str),
            color="tier", orientation="h",
            color_discrete_map={"Top 10": "#2563EB", "Bottom 10": "#EF4444"},
            labels={"total_sales": "Total Sales", "y": "Item ID"},
        )
        fig_i.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig_i, use_container_width=True)

# ----------------- Tab 4: Stationarity & ACF -----------------
with tab4:
    st.subheader("Stationarity Tests & Autocorrelation Diagnostics")

    # Load pre-computed stationarity report
    stat_file = Path("reports/data_quality/stationarity_diagnostics.json")
    if stat_file.exists():
        with open(stat_file, "r", encoding="utf-8") as f:
            stat_diag = json.load(f)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Total Network Demand")
            res_agg = stat_diag["overall_aggregate_demand"]
            st.info(f"**Conclusion**: {res_agg['conclusion']}\n\n- ADF Stat: `{res_agg['adf']['statistic']}` (p={res_agg['adf']['p_value']})\n- KPSS Stat: `{res_agg['kpss']['statistic']}` (p={res_agg['kpss']['p_value']})")
        with c2:
            st.markdown("#### Store 1, Item 1 (Raw)")
            res_s1 = stat_diag["sample_series_store1_item1"]
            st.info(f"**Conclusion**: {res_s1['conclusion']}\n\n- ADF Stat: `{res_s1['adf']['statistic']}` (p={res_s1['adf']['p_value']})\n- KPSS Stat: `{res_s1['kpss']['statistic']}` (p={res_s1['kpss']['p_value']})")
        with c3:
            st.markdown("#### Store 1, Item 1 (1st Difference)")
            res_diff = stat_diag["sample_series_store1_item1_diff_1"]
            st.success(f"**Conclusion**: {res_diff['conclusion']}\n\n- ADF Stat: `{res_diff['adf']['statistic']}` (p={res_diff['adf']['p_value']})\n- KPSS Stat: `{res_diff['kpss']['statistic']}` (p={res_diff['kpss']['p_value']})")

    st.markdown("#### Autocorrelation & Partial Autocorrelation (Store 1, Item 1)")
    s1i1 = df[(df["store"] == 1) & (df["item"] == 1)].sort_values("date").set_index("date")["sales"]
    acf_pacf_res = compute_acf_pacf(s1i1, nlags=28)

    acf_col, pacf_col = st.columns(2)
    with acf_col:
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=acf_pacf_res["lags"], y=acf_pacf_res["acf"], marker_color="#2563EB", width=0.4))
        fig_acf.add_trace(go.Scatter(x=acf_pacf_res["lags"], y=acf_pacf_res["acf_confint_upper"], mode="lines", line=dict(color="#EF4444", dash="dash"), showlegend=False))
        fig_acf.add_trace(go.Scatter(x=acf_pacf_res["lags"], y=acf_pacf_res["acf_confint_lower"], mode="lines", line=dict(color="#EF4444", dash="dash"), showlegend=False))
        fig_acf.update_layout(title="Autocorrelation Function (ACF) — Strong 7, 14, 21, 28-day lags", template="plotly_white", height=350, xaxis_title="Lag (Days)", yaxis_title="Correlation")
        st.plotly_chart(fig_acf, use_container_width=True)

    with pacf_col:
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(x=acf_pacf_res["lags"], y=acf_pacf_res["pacf"], marker_color="#10B981", width=0.4))
        fig_pacf.add_trace(go.Scatter(x=acf_pacf_res["lags"], y=acf_pacf_res["pacf_confint_upper"], mode="lines", line=dict(color="#EF4444", dash="dash"), showlegend=False))
        fig_pacf.add_trace(go.Scatter(x=acf_pacf_res["lags"], y=acf_pacf_res["pacf_confint_lower"], mode="lines", line=dict(color="#EF4444", dash="dash"), showlegend=False))
        fig_pacf.update_layout(title="Partial Autocorrelation Function (PACF)", template="plotly_white", height=350, xaxis_title="Lag (Days)", yaxis_title="Partial Correlation")
        st.plotly_chart(fig_pacf, use_container_width=True)
