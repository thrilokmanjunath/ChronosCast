"""Unit tests for feature engineering and leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronoscast.features.direct_horizon import create_direct_horizon_dataset
from chronoscast.features.pipeline import TimeSeriesFeaturePipeline


@pytest.fixture
def sample_multiseries_df() -> pd.DataFrame:
    """Fixture with 2 stores, 2 items, 40 daily records each."""
    dates = pd.date_range("2016-01-01", periods=40, freq="D")
    records = []
    for s in [1, 2]:
        for i in [1, 2]:
            for idx, d in enumerate(dates):
                # Series has distinct base value so cross-contamination is immediately obvious
                sales = 100 * s + 10 * i + idx
                records.append({
                    "date": d,
                    "store": s,
                    "item": i,
                    "sales": float(sales),
                })
    return pd.DataFrame(records)


def test_lag_alignment_and_no_leakage(sample_multiseries_df: pd.DataFrame):
    pipeline = TimeSeriesFeaturePipeline(lags=[1, 7], rolling_means=[7], rolling_stds=[7])
    pipeline.fit(sample_multiseries_df)
    transformed = pipeline.transform(sample_multiseries_df)

    # Filter to Store 1, Item 1
    s1i1 = transformed[(transformed["store"] == 1) & (transformed["item"] == 1)].sort_values("date").reset_index(drop=True)

    # Day 0: lag_1 must be NaN
    assert pd.isna(s1i1.loc[0, "sales_lag_1"])

    # Day 1: lag_1 must be exactly Day 0 sales
    assert s1i1.loc[1, "sales_lag_1"] == s1i1.loc[0, "sales"]

    # Day 7: lag_7 must be exactly Day 0 sales
    assert s1i1.loc[7, "sales_lag_7"] == s1i1.loc[0, "sales"]

    # Rolling mean at Day 7: must be the mean of Day 0 to Day 6 sales (7 values).
    # It must NOT include Day 7 sales!
    expected_rolling_mean_7 = s1i1.loc[0:6, "sales"].mean()
    assert np.isclose(s1i1.loc[7, "rolling_mean_7"], expected_rolling_mean_7)


def test_entity_boundary_isolation(sample_multiseries_df: pd.DataFrame):
    pipeline = TimeSeriesFeaturePipeline(lags=[1, 7], rolling_means=[7])
    pipeline.fit(sample_multiseries_df)
    transformed = pipeline.transform(sample_multiseries_df)

    # Store 2 Item 1 at day 0 should NOT inherit lag_1 from Store 1 Item 2 at day 39!
    s2i1 = transformed[(transformed["store"] == 2) & (transformed["item"] == 1)].sort_values("date").reset_index(drop=True)
    assert pd.isna(s2i1.loc[0, "sales_lag_1"])
    assert s2i1.loc[1, "sales_lag_1"] == s2i1.loc[0, "sales"]
    assert s2i1.loc[0, "sales"] >= 210  # from 100*2 + 10*1


def test_direct_horizon_alignment(sample_multiseries_df: pd.DataFrame):
    pipeline = TimeSeriesFeaturePipeline(lags=[1, 7], rolling_means=[7])
    pipeline.fit(sample_multiseries_df)
    transformed = pipeline.transform(sample_multiseries_df)

    df_h7, feature_cols = create_direct_horizon_dataset(transformed, horizon=7, pipeline=pipeline)
    assert "target_h7" in df_h7.columns

    s1i1 = df_h7[(df_h7["store"] == 1) & (df_h7["item"] == 1)].sort_values("date").reset_index(drop=True)
    # At index 0, target_h7 must equal sales at index 7
    full_s1i1 = sample_multiseries_df[(sample_multiseries_df["store"] == 1) & (sample_multiseries_df["item"] == 1)].sort_values("date").reset_index(drop=True)
    assert s1i1.loc[0, "target_h7"] == full_s1i1.loc[7, "sales"]
