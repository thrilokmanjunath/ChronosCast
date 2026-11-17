"""Pydantic schemas for ChronosCast FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Service health state")
    app_name: str = Field(default="ChronosCast API")
    version: str = Field(default="0.1.0")
    timestamp: str


class DatasetSummaryResponse(BaseModel):
    total_sales: float
    mean_total_daily_sales: float
    store_count: int
    item_count: int
    start_date: str
    end_date: str
    total_days: int
    total_observations: int


class StoreInfo(BaseModel):
    store: int
    total_sales: float
    mean_sales: float
    std_sales: float
    sales_share_pct: float


class ItemInfo(BaseModel):
    item: int
    total_sales: float
    mean_sales: float
    std_sales: float
    sales_share_pct: float


class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    is_primary: bool
    status: str


class ForecastRequest(BaseModel):
    store: Optional[int] = Field(None, ge=1, le=10, description="Store ID (1 to 10), or None for store aggregation")
    item: Optional[int] = Field(None, ge=1, le=50, description="Item ID (1 to 50), or None for item aggregation")
    horizon: int = Field(default=7, description="Forecast horizon in days (7, 14, or 28)")
    confidence_level: float = Field(default=0.90, description="Prediction interval confidence (0.80, 0.90, 0.95)")
    model: Optional[str] = Field(default="lightgbm", description="Forecasting model to use")


class ForecastResponse(BaseModel):
    status: str
    model: str
    aggregation_type: str
    series_title: str
    store: Optional[int]
    item: Optional[int]
    horizon: int
    confidence_level: float
    generated_at: str
    dates: List[str]
    predictions: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    forecast_total: float


class HistoryRequest(BaseModel):
    store: Optional[int] = Field(None, ge=1, le=10)
    item: Optional[int] = Field(None, ge=1, le=50)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PerformanceResponse(BaseModel):
    leaderboard: List[Dict[str, Any]]
    test_report: Dict[str, Any]
    error_breakdown: Dict[str, Any]
