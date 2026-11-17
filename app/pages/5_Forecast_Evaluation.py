"""ChronosCast — Forecast Evaluation & Residual Diagnostics Page."""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.utils import (
    apply_custom_styles,
    load_error_breakdown,
    load_test_report,
    render_sidebar_header,
)

st.set_page_config(page_title="Forecast Evaluation — ChronosCast", page_icon="📉", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("📉 Forecast Evaluation & Diagnostics")
st.markdown("In-depth evaluation on the held-out final test set (2017 H2): residual analysis, slice errors, and feature importance.")

test_rep = load_test_report()
breakdown = load_error_breakdown()

# Top metric summary
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Test WAPE", f"{test_rep['overall_metrics']['wape']}%")
with m2:
    st.metric("Test MAE", f"{test_rep['overall_metrics']['mae']:.2f} Units")
with m3:
    st.metric("Test RMSE", f"{test_rep['overall_metrics']['rmse']:.2f} Units")
with m4:
    st.metric("Test Observations", f"{test_rep['test_observations']:,}")

st.markdown("---")

# Row 1: Conformal Coverage & Demand Tiers
col_cov, col_tier = st.columns(2)

with col_cov:
    st.subheader("🛡️ Prediction Interval Coverage (Held-out Test)")
    cov_rows = []
    for conf, stats in test_rep["uncertainty_coverage"].items():
        cov_rows.append({
            "Confidence Level": conf,
            "Nominal Target": f"{stats['nominal_confidence_pct']}%",
            "Empirical Test Coverage": f"{stats['empirical_coverage_pct']}%",
            "Margin of Error": f"±{stats['margin_of_error']} Units",
            "Mean Interval Width": f"{stats['mean_interval_width']} Units",
        })
    st.dataframe(pd.DataFrame(cov_rows), hide_index=True, use_container_width=True)
    st.caption("Empirical coverage closely matches nominal targets without future leakage.")

with col_tier:
    st.subheader("📦 High-Demand vs Low-Demand Slices")
    tier_df = pd.DataFrame([
        {"Tier": "High-Demand SKUs (Top 50%)", "WAPE (%)": breakdown["high_demand_tier"]["wape"], "MAE": breakdown["high_demand_tier"]["mae"], "sMAPE (%)": breakdown["high_demand_tier"]["smape"]},
        {"Tier": "Low-Demand SKUs (Bottom 50%)", "WAPE (%)": breakdown["low_demand_tier"]["wape"], "MAE": breakdown["low_demand_tier"]["mae"], "sMAPE (%)": breakdown["low_demand_tier"]["smape"]},
    ])
    st.dataframe(tier_df, hide_index=True, use_container_width=True)
    st.caption("Lower absolute error on low-volume items; relative WAPE remains balanced across tiers.")

st.markdown("---")

# Row 2: Store Breakdown & Feature Importance
col_store_err, col_feat = st.columns(2)

with col_store_err:
    st.subheader("🏪 Error Breakdown by Store (Test WAPE %)")
    store_err_df = pd.DataFrame(breakdown["by_store"])
    fig_se = px.bar(
        store_err_df,
        x=store_err_df["store"].astype(str),
        y="wape",
        color="wape",
        color_continuous_scale="Blues_r",
        text="wape",
        labels={"x": "Store ID", "wape": "WAPE (%)"},
    )
    fig_se.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_se.update_layout(template="plotly_white", height=380, showlegend=False)
    st.plotly_chart(fig_se, use_container_width=True)

with col_feat:
    st.subheader("🌟 Top 15 Feature Importances (LightGBM)")
    feat_df = pd.DataFrame(
        list(breakdown["feature_importances"].items()),
        columns=["Feature", "Normalized Importance"],
    ).sort_values("Normalized Importance", ascending=True)

    fig_f = px.bar(
        feat_df,
        x="Normalized Importance",
        y="Feature",
        orientation="h",
        color="Normalized Importance",
        color_continuous_scale="Viridis",
    )
    fig_f.update_layout(template="plotly_white", height=380, showlegend=False)
    st.plotly_chart(fig_f, use_container_width=True)
