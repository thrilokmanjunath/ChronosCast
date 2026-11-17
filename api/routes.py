"""FastAPI route handlers for ChronosCast."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status

from api.schemas import (
    DatasetSummaryResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ItemInfo,
    ModelInfo,
    PerformanceResponse,
    StoreInfo,
)
from chronoscast.forecasting.engine import ForecastingEngine
from chronoscast.models.registry import list_available_models
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1")

# Global singleton engine
_engine: Optional[ForecastingEngine] = None


def get_engine() -> ForecastingEngine:
    global _engine
    if _engine is None:
        _engine = ForecastingEngine()
    return _engine


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return platform operational status and timestamp."""
    return HealthResponse(
        status="healthy",
        app_name="ChronosCast API",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/datasets/summary", response_model=DatasetSummaryResponse, tags=["Data"])
def get_dataset_summary() -> DatasetSummaryResponse:
    """Get overall dataset descriptive summary."""
    summary_path = Path("reports/data_quality/eda_summary.json")
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="EDA summary not found. Run EDA pipeline first.")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetSummaryResponse(**data)


@router.get("/stores", response_model=List[StoreInfo], tags=["Data"])
def list_stores() -> List[StoreInfo]:
    """Retrieve list of all 10 stores and aggregate sales statistics."""
    csv_path = Path("reports/data_quality/store_summary.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Store summary not found.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@router.get("/items", response_model=List[ItemInfo], tags=["Data"])
def list_items() -> List[ItemInfo]:
    """Retrieve list of all 50 items and demand velocity statistics."""
    csv_path = Path("reports/data_quality/item_summary.csv")
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Item summary not found.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@router.get("/models", response_model=List[ModelInfo], tags=["Models"])
def list_models() -> List[ModelInfo]:
    """List registered forecasting models and active deployment status."""
    available = list_available_models()
    models = []
    for m in available:
        is_primary = m == "lightgbm"
        models.append(ModelInfo(
            model_name=m,
            model_type="Machine Learning" if m in ["lightgbm", "random_forest", "xgboost"] else "Statistical/Baseline",
            is_primary=is_primary,
            status="active" if is_primary or m in ["naive", "seasonal_naive", "moving_average"] else "registered",
        ))
    return models


@router.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def generate_forecast(req: ForecastRequest) -> ForecastResponse:
    """Generate sales forecast and prediction intervals dynamically."""
    if req.horizon not in [7, 14, 28]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported horizon {req.horizon}. Standard horizons are 7, 14, or 28 days.",
        )
    if req.confidence_level not in [0.80, 0.90, 0.95]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported confidence level {req.confidence_level}. Supported: [0.80, 0.90, 0.95].",
        )

    engine = get_engine()
    try:
        res = engine.generate_forecast(
            store=req.store,
            item=req.item,
            horizon=req.horizon,
            confidence_level=req.confidence_level,
            model_name=req.model,
        )
        return ForecastResponse(**res)
    except Exception as e:
        logger.error("Forecast generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/forecast/history", tags=["Forecasting"])
def get_forecast_history(
    store: Optional[int] = Query(None, ge=1, le=10),
    item: Optional[int] = Query(None, ge=1, le=50),
    days: int = Query(60, ge=7, le=365),
):
    """Retrieve recent actual demand history for context."""
    path = Path("data/interim/cleaned_sales.parquet")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cleaned data not found.")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if store is not None:
        df = df[df["store"] == store]
    if item is not None:
        df = df[df["item"] == item]

    last_date = df["date"].max()
    start_date = last_date - pd.Timedelta(days=days)
    recent = df[df["date"] >= start_date].groupby("date")["sales"].sum().reset_index()

    return {
        "store": store,
        "item": item,
        "days": days,
        "dates": [str(d.date()) for d in recent["date"]],
        "sales": recent["sales"].tolist(),
    }


@router.get("/model/performance", response_model=PerformanceResponse, tags=["Evaluation"])
def get_model_performance() -> PerformanceResponse:
    """Retrieve benchmark leaderboard and test evaluation metrics."""
    metrics_dir = Path("artifacts/metrics")
    lb_path = metrics_dir / "leaderboard.json"
    rep_path = metrics_dir / "test_evaluation_report.json"
    err_path = metrics_dir / "error_breakdown.json"

    if not (lb_path.exists() and rep_path.exists() and err_path.exists()):
        raise HTTPException(status_code=404, detail="Evaluation metrics not found. Run model training first.")

    with open(lb_path, "r", encoding="utf-8") as f:
        leaderboard = json.load(f)
    with open(rep_path, "r", encoding="utf-8") as f:
        test_report = json.load(f)
    with open(err_path, "r", encoding="utf-8") as f:
        error_breakdown = json.load(f)

    return PerformanceResponse(
        leaderboard=leaderboard,
        test_report=test_report,
        error_breakdown=error_breakdown,
    )
