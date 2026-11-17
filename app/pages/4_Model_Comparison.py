"""ChronosCast — Model Benchmarking & Comparison Page."""

import plotly.express as px
import streamlit as st

from app.utils import apply_custom_styles, load_leaderboard, render_sidebar_header

st.set_page_config(page_title="Model Benchmarking — ChronosCast", page_icon="🏆", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("🏆 Model Benchmarking & Leaderboard")
st.markdown("Rigorous chronological validation leaderboard comparing Baselines, Classical Statistical methods, and Machine Learning models.")

leaderboard = load_leaderboard()

# Display formatted table
st.subheader("Official Validation Leaderboard (H1 2017 Validation Window)")
st.dataframe(
    leaderboard[[
        "rank", "model", "model_type", "wape", "mae", "rmse", "smape", "mape", "runtime_sec"
    ]].rename(columns={
        "rank": "Rank",
        "model": "Model Name",
        "model_type": "Category",
        "wape": "WAPE (%)",
        "mae": "MAE",
        "rmse": "RMSE",
        "smape": "sMAPE (%)",
        "mape": "MAPE (%)",
        "runtime_sec": "Runtime (s)",
    }),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# Visual comparisons
col_wape, col_time = st.columns(2)

with col_wape:
    st.subheader("Forecast Error Comparison (WAPE %)")
    fig_wape = px.bar(
        leaderboard.sort_values("wape"),
        x="wape",
        y="model",
        orientation="h",
        color="model_type",
        text="wape",
        color_discrete_map={
            "Machine Learning": "#2563EB",
            "Statistical": "#10B981",
            "Baseline": "#94A3B8",
        },
        labels={"wape": "Weighted Absolute Percentage Error (WAPE %)", "model": "Model"},
    )
    fig_wape.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_wape.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_wape, use_container_width=True)

with col_time:
    st.subheader("Compute Efficiency & Runtime (Seconds)")
    fig_time = px.bar(
        leaderboard.sort_values("runtime_sec"),
        x="runtime_sec",
        y="model",
        orientation="h",
        color="model_type",
        text="runtime_sec",
        color_discrete_map={
            "Machine Learning": "#2563EB",
            "Statistical": "#10B981",
            "Baseline": "#94A3B8",
        },
        labels={"runtime_sec": "Training & Inference Time (Seconds)", "model": "Model"},
    )
    fig_time.update_traces(texttemplate="%{text:.2f}s", textposition="outside")
    fig_time.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_time, use_container_width=True)

# Technical Discussion
st.markdown("---")
st.subheader("Analytical Insights & Methodological Findings")
st.markdown("""
1. **Machine Learning Supremacy**: The global pooled **LightGBM Regressor** achieved top rank (**10.53% WAPE**, **5.92 MAE**), capturing cross-series seasonal interactions, item velocity tiers, and annual growth that independent univariate models cannot pool across 500 series.
2. **Competitive XGBoost Execution**: XGBoost followed closely at **10.71% WAPE** with rapid histogram boosting.
3. **Strong Seasonal Baseline**: **Seasonal Naive ($s=7$)** achieved **18.23% WAPE**, outperforming pure persistence (**26.58%**), confirming that weekly day-of-week seasonality is the dominant recurring rhythm.
4. **Holt-Winters Scalability Constraint**: While Holt-Winters achieves strong local fit (**17.32% WAPE**), fitting 500 individual exponential smoothing models sequentially incurs heavy runtime overhead compared to one vectorized global LightGBM model.
""")
