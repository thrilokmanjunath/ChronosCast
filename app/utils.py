"""Shared UI components, theming, and caching utilities for Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")


def apply_custom_styles() -> None:
    """Inject custom modern CSS styling."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }
        .metric-card h3 {
            font-size: 0.85rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0;
        }
        .metric-card p {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0F172A;
            margin: 8px 0 0 0;
        }
        .stButton>button {
            background-color: #2563EB;
            color: white;
            border-radius: 6px;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_header() -> None:
    """Render consistent sidebar branding."""
    with st.sidebar:
        st.markdown("### ⏱️ ChronosCast")
        st.caption("Intelligent Time Series Analytics & Forecasting Platform")
        st.markdown("---")


@st.cache_data
def load_interim_data() -> pd.DataFrame:
    """Load cleaned parquet sales data."""
    df = pd.read_parquet("data/interim/cleaned_sales.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_eda_summary() -> Dict[str, Any]:
    with open("reports/data_quality/eda_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    return pd.read_csv("artifacts/metrics/leaderboard.csv")


@st.cache_data
def load_test_report() -> Dict[str, Any]:
    with open("artifacts/metrics/test_evaluation_report.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_error_breakdown() -> Dict[str, Any]:
    with open("artifacts/metrics/error_breakdown.json", "r", encoding="utf-8") as f:
        return json.load(f)
