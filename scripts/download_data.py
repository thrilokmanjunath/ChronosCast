"""Data acquisition script for ChronosCast.

Downloads and verifies the Kaggle Store Item Demand Forecasting dataset.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_URL = "https://raw.githubusercontent.com/sanjojoys/Sales_Forecasting_TimeSeries/main/data/sales_data.csv"
EXPECTED_ROWS = 913_000
EXPECTED_HEADER = "date,store,item,sales"


def acquire_data(output_path: str | Path = "data/raw/train.csv") -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        logger.info("File %s already exists. Checking line count...", out)
        with open(out, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        if lines == EXPECTED_ROWS + 1:
            logger.info("Existing file has valid line count (%d). Skipping download.", lines)
            return out
        logger.warning("Existing file has invalid line count (%d). Re-downloading...", lines)

    logger.info("Downloading raw dataset from %s to %s...", RAW_DATA_URL, out)
    urllib.request.urlretrieve(RAW_DATA_URL, out)

    with open(out, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        lines = 1 + sum(1 for _ in f)

    if first_line != EXPECTED_HEADER:
        raise ValueError(f"Unexpected header: {first_line}. Expected: {EXPECTED_HEADER}")

    if lines != EXPECTED_ROWS + 1:
        raise ValueError(f"Unexpected line count: {lines}. Expected: {EXPECTED_ROWS + 1}")

    logger.info("Successfully acquired %s with %d rows.", out, lines - 1)
    return out


if __name__ == "__main__":
    acquire_data()
