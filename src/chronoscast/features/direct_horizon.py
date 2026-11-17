"""Direct multi-horizon feature alignment for ChronosCast.

Prepares train/validation/test matrices where feature vectors at time T
predict target at T + horizon (h in {7, 14, 28}) without temporal leakage.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from chronoscast.features.pipeline import TimeSeriesFeaturePipeline
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_direct_horizon_dataset(
    df: pd.DataFrame,
    horizon: int,
    pipeline: TimeSeriesFeaturePipeline,
    target_col: str = "sales",
    store_col: str = "store",
    item_col: str = "item",
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target vector y for direct horizon h forecasting.

    At observation date t, the target is y_{t + horizon}.
    Features X_t are calculated strictly using data up to date t.
    Zero target leakage across the horizon boundary.

    Args:
        df: Enriched feature DataFrame (output of TimeSeriesFeaturePipeline.transform).
        horizon: Forecast lead steps in days (e.g., 7, 14, 28).
        pipeline: The feature pipeline instance used.
        target_col: Target variable name.
        store_col: Store identifier column.
        item_col: Item identifier column.
        date_col: Date column.

    Returns:
        Tuple of (X DataFrame, y Series, alignment_metadata DataFrame).
    """
    logger.info("Aligning direct forecasting dataset for horizon = %d days...", horizon)

    # Sort strictly by entity and date
    df_sorted = df.sort_values(by=[store_col, item_col, date_col]).copy()

    # Shift target backward by horizon within each series to create y_{t + horizon}
    # i.e., lead(target, horizon)
    grouped = df_sorted.groupby([store_col, item_col], observed=True)[target_col]
    lead_target = grouped.shift(-horizon)

    target_name = f"target_h{horizon}"
    df_sorted[target_name] = lead_target

    # Drop rows where target is NaN (the last horizon days of each series)
    valid_mask = df_sorted[target_name].notna()
    df_valid = df_sorted[valid_mask].copy()

    # Feature columns
    feature_cols = [c for c in pipeline.get_feature_names() if c in df_valid.columns]

    return df_valid, feature_cols
