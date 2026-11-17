"""Build and cache the full engineered feature dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chronoscast.features.pipeline import TimeSeriesFeaturePipeline
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_and_save_features(
    input_path: str = "data/interim/cleaned_sales.parquet",
    output_path: str = "data/processed/features_sales.parquet",
) -> pd.DataFrame:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading cleaned data from %s...", input_path)
    df = pd.read_parquet(input_path)

    pipeline = TimeSeriesFeaturePipeline()
    pipeline.fit(df)

    logger.info("Engineering features across 500 store-item series...")
    df_features = pipeline.transform(df)

    logger.info("Writing processed feature dataset to %s...", out)
    df_features.to_parquet(out, index=False, engine="pyarrow")
    logger.info("Successfully produced %s with %d rows and %d columns.", out, len(df_features), df_features.shape[1])
    logger.info("Feature columns: %s", pipeline.get_feature_names())

    return df_features


if __name__ == "__main__":
    build_and_save_features()
