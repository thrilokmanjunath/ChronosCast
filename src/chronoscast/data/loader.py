"""Data loading module for ChronosCast."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)

EXPECTED_COLUMNS = ["date", "store", "item", "sales"]


def load_raw_data(
    file_path: str | Path = "data/raw/train.csv",
    parse_dates: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw store item demand data from CSV.

    Args:
        file_path: Path to raw train.csv.
        parse_dates: Whether to parse date column as datetime.
        nrows: Optional number of rows to read.

    Returns:
        DataFrame containing raw data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(
        path,
        parse_dates=["date"] if parse_dates else False,
        nrows=nrows,
    )

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in raw data: {missing_cols}")

    logger.info("Loaded %d rows with columns: %s", len(df), list(df.columns))
    return df
