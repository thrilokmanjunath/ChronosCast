"""ChronosCast Streamlit Application — Home & Executive Overview."""

import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    apply_custom_styles,
    load_eda_summary,
    load_interim_data,
    load_test_report,
    render_sidebar_header,
)

st.set_page_config(
    page_title="ChronosCast — Executive Overview",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()
render_sidebar_header()

st.markdown('<h1 class="main-header">ChronosCast Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Time Series Trend Analysis, Forecasting & Inventory Planning System</p>', unsafe_allow_html=True)

# Load actual processed data and summary metrics
eda_sum = load_eda_summary()
test_rep = load_test_report()
df = load_interim_data()

# Top KPI row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        label="Total Sales Volume",
        value=f"{eda_sum['total_sales'] / 1e6:.1f}M Units",
        help="Aggregated units sold across all 10 stores and 50 products (2013-2017)",
    )
with col2:
    st.metric(
        label="Network Stores",
        value=eda_sum["store_count"],
        help="Retail store locations",
    )
with col3:
    st.metric(
        label="Product Catalog",
        value=f"{eda_sum['item_count']} SKUs",
        help="Unique product items monitored",
    )
with col4:
    st.metric(
        label="Time Horizon",
        value=f"{eda_sum['total_days']} Days",
        help="Complete daily observation calendar: 2013-01-01 to 2017-12-31",
    )
with col5:
    st.metric(
        label="Production Model WAPE",
        value=f"{test_rep['overall_metrics']['wape']}%",
        delta="-7.25% vs Baseline",
        help="Weighted Absolute Percentage Error on held-out 2017 H2 Test Set",
    )

st.markdown("---")

# Layout: High-level Trajectory + Architecture Brief
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📈 Multi-Year Macro Demand Trajectory")
    daily = df.groupby("date")["sales"].sum().reset_index()
    daily["rolling_30"] = daily["sales"].rolling(30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["sales"],
        mode="lines", name="Daily Total Sales",
        line=dict(color="#CBD5E1", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling_30"],
        mode="lines", name="30-Day Moving Average",
        line=dict(color="#2563EB", width=2.5),
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=380,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("🎯 System Status & Performance")
    st.markdown(f"""
    - **Active Model**: `{test_rep['best_model']}`
    - **Evaluation Strategy**: Leakage-free Chronological Split
    - **Test Window**: `{test_rep['test_period']}`
    - **Test MAE**: `{test_rep['overall_metrics']['mae']:.2f}` units
    - **Test RMSE**: `{test_rep['overall_metrics']['rmse']:.2f}` units
    - **Test sMAPE**: `{test_rep['overall_metrics']['smape']:.2f}%`
    - **90% Interval Coverage**: `{test_rep['uncertainty_coverage']['90%']['empirical_coverage_pct']}%` (Target: 90%)
    """)
    st.info("💡 **Navigation Guide**: Explore historical patterns in **Data Explorer**, inspect periodicity in **Trend Analysis**, or produce out-of-sample projections in **Forecast Studio**.")

st.markdown("---")
st.subheader("Platform Capabilities")
cap1, cap2, cap3 = st.columns(3)
with cap1:
    st.markdown("#### 1. End-to-End Temporal Modeling")
    st.write("Strictly chronological 2013-2016 training with H1 2017 validation and untouched H2 2017 test evaluation. Zero future target leakage.")
with cap2:
    st.markdown("#### 2. Conformal Prediction Intervals")
    st.write("Residual quantile calibration providing 80%, 90%, and 95% non-parametric confidence bands validated on test data.")
with cap3:
    st.markdown("#### 3. Prescriptive Inventory Planning")
    st.write("Dynamic reorder points, safety stock recommendations, and stockout risk indicators powered by multi-horizon demand distributions.")
