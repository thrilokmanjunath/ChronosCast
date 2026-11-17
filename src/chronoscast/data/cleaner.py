"""Data cleaning and preparation module for ChronosCast."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chronoscast.data.loader import load_raw_data
from chronoscast.data.validator import DataQualityValidator
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_and_prepare_data(
    raw_path: str | Path = "data/raw/train.csv",
    output_path: str | Path = "data/interim/cleaned_sales.parquet",
    report_json_path: str | Path = "reports/data_quality/data_quality_report.json",
    report_md_path: str | Path = "reports/data_quality/data_quality_report.md",
) -> pd.DataFrame:
    """Validate, clean, sort, and serialize interim dataset.

    Args:
        raw_path: Path to raw train.csv.
        output_path: Path to save cleaned parquet dataset.
        report_json_path: Path to write validation report JSON.
        report_md_path: Path to write human-readable markdown report.

    Returns:
        Cleaned, sorted DataFrame.
    """
    df = load_raw_data(raw_path)

    # Run validation
    validator = DataQualityValidator()
    report = validator.validate(df)

    # Save reports
    Path(report_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Build Markdown report
    md_content = f"""# ChronosCast — Data Quality Report

## Executive Summary
- **Validation Status**: {'✅ PASSED' if report['passed'] else '❌ FAILED'}
- **Total Observations**: {report['total_rows']:,}
- **Unique Stores**: {report['entity_counts']['unique_stores']} (IDs 1-10)
- **Unique Items**: {report['entity_counts']['unique_items']} (IDs 1-50)
- **Total Time Series**: {report['entity_counts']['total_series']}
- **Date Range**: {report['date_range']['min_date']} to {report['date_range']['max_date']} ({report['date_range']['unique_days']} calendar days)

## Quality Checklist
| Check | Status | Details |
|---|---|---|
| No Missing Values | {'✅ Pass' if report['checks']['no_missing_values'] else '❌ Fail'} | Total NaNs = {sum(report['null_counts'].values())} |
| No Duplicate Keys | {'✅ Pass' if report['checks']['no_duplicate_keys'] else '❌ Fail'} | Duplicates = {report['duplicate_keys']} |
| Full Date Coverage | {'✅ Pass' if report['checks']['date_range_valid'] else '❌ Fail'} | 2013-01-01 to 2017-12-31 |
| All Days Present | {'✅ Pass' if report['checks']['all_calendar_days_present'] else '❌ Fail'} | {report['date_range']['unique_days']} of {report['date_range']['expected_days']} days |
| Store IDs [1-10] | {'✅ Pass' if report['checks']['store_identifiers_valid'] else '❌ Fail'} | Stores 1 to 10 verified |
| Item IDs [1-50] | {'✅ Pass' if report['checks']['item_identifiers_valid'] else '❌ Fail'} | Items 1 to 50 verified |
| Full Grid Completeness | {'✅ Pass' if report['checks']['full_temporal_grid_intact'] else '❌ Fail'} | Exactly {report['expected_total_rows']:,} records ($10 \\times 50 \\times 1826$) |
| Non-Negative Sales | {'✅ Pass' if report['checks']['non_negative_sales'] else '❌ Fail'} | Min sales = {report['sales_distribution']['min']} |

## Sales Distribution
- **Minimum**: {report['sales_distribution']['min']}
- **Maximum**: {report['sales_distribution']['max']}
- **Mean**: {report['sales_distribution']['mean']}
- **Standard Deviation**: {report['sales_distribution']['std']}
- **Median**: {report['sales_distribution']['median']}
"""
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    if not report["passed"]:
        raise ValueError("Data validation failed. See data_quality_report.json for details.")

    # Clean and cast types
    df["date"] = pd.to_datetime(df["date"])
    df["store"] = df["store"].astype("int32")
    df["item"] = df["item"].astype("int32")
    df["sales"] = df["sales"].astype("float32")

    # Sort strictly by entity and date
    logger.info("Sorting observations by (store, item, date)...")
    df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)

    # Save to parquet
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Successfully written cleaned interim dataset to: %s (%d rows)", out_path, len(df))

    return df


if __name__ == "__main__":
    clean_and_prepare_data()
