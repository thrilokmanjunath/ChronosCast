"""Unit tests for data loading and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from chronoscast.data.validator import DataQualityValidator


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    """Create a minimal valid synthetic dataset fixture."""
    dates = pd.date_range("2013-01-01", periods=14, freq="D")
    records = []
    for store in [1, 2]:
        for item in [1, 2]:
            for d in dates:
                records.append({
                    "date": d,
                    "store": store,
                    "item": item,
                    "sales": 10 + store + item,
                })
    return pd.DataFrame(records)


def test_validator_passes_on_valid_data(sample_sales_df: pd.DataFrame):
    validator = DataQualityValidator(
        expected_stores=2,
        expected_items=2,
        expected_start_date="2013-01-01",
        expected_end_date="2013-01-14",
    )
    report = validator.validate(sample_sales_df)
    assert report["passed"] is True
    assert report["checks"]["no_missing_values"] is True
    assert report["checks"]["no_duplicate_keys"] is True
    assert report["checks"]["non_negative_sales"] is True


def test_validator_fails_on_negative_sales(sample_sales_df: pd.DataFrame):
    df_invalid = sample_sales_df.copy()
    df_invalid.loc[0, "sales"] = -5
    validator = DataQualityValidator(
        expected_stores=2,
        expected_items=2,
        expected_start_date="2013-01-01",
        expected_end_date="2013-01-14",
    )
    report = validator.validate(df_invalid)
    assert report["passed"] is False
    assert report["checks"]["non_negative_sales"] is False


def test_validator_fails_on_duplicates(sample_sales_df: pd.DataFrame):
    df_dup = pd.concat([sample_sales_df, sample_sales_df.iloc[[0]]], ignore_index=True)
    validator = DataQualityValidator(
        expected_stores=2,
        expected_items=2,
        expected_start_date="2013-01-01",
        expected_end_date="2013-01-14",
    )
    report = validator.validate(df_dup)
    assert report["passed"] is False
    assert report["checks"]["no_duplicate_keys"] is False


def test_interim_cleaned_dataset():
    df = pd.read_parquet("data/interim/cleaned_sales.parquet")
    assert len(df) == 913_000
    assert set(df.columns) == {"date", "store", "item", "sales"}
    assert df["store"].nunique() == 10
    assert df["item"].nunique() == 50
    assert df["sales"].min() >= 0
