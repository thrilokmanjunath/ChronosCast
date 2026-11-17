"""ChronosCast — Reports & Artifact Exports Page."""

from pathlib import Path

import streamlit as st

from app.utils import apply_custom_styles, render_sidebar_header

st.set_page_config(page_title="Reports & Exports — ChronosCast", page_icon="📑", layout="wide")
apply_custom_styles()
render_sidebar_header()

st.title("📑 Technical Reports & Artifact Exports")
st.markdown("Download official benchmark metrics, data quality audits, and diagnostic evaluation reports.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Benchmark Leaderboard")
    lb_path = Path("artifacts/metrics/leaderboard.csv")
    if lb_path.exists():
        with open(lb_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download Official Leaderboard (CSV)",
                data=f.read(),
                file_name="chronoscast_official_leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.subheader("🛡️ Data Quality & Validation Audit")
    dq_md = Path("reports/data_quality/data_quality_report.md")
    if dq_md.exists():
        with open(dq_md, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download Data Quality Report (Markdown)",
                data=f.read(),
                file_name="chronoscast_data_quality_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

with col2:
    st.subheader("🎯 Test Set Evaluation & Error Breakdown")
    rep_json = Path("artifacts/metrics/test_evaluation_report.json")
    if rep_json.exists():
        with open(rep_json, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download Test Set Evaluation Report (JSON)",
                data=f.read(),
                file_name="chronoscast_test_evaluation_report.json",
                mime="application/json",
                use_container_width=True,
            )

    st.subheader("🔬 Stationarity Diagnostics (ADF & KPSS)")
    stat_json = Path("reports/data_quality/stationarity_diagnostics.json")
    if stat_json.exists():
        with open(stat_json, "r", encoding="utf-8") as f:
            st.download_button(
                label="📥 Download Stationarity Diagnostics (JSON)",
                data=f.read(),
                file_name="chronoscast_stationarity_diagnostics.json",
                mime="application/json",
                use_container_width=True,
            )

st.markdown("---")
st.subheader("Data Quality Report Preview")
if dq_md.exists():
    with open(dq_md, "r", encoding="utf-8") as f:
        st.markdown(f.read())
