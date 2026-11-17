"""Model registry and factory for ChronosCast."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from chronoscast.models.base import BaseForecaster
from chronoscast.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from chronoscast.models.ml_forecaster import (
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)
from chronoscast.models.statistical import ARIMAForecaster, ExponentialSmoothingForecaster
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_REGISTRY: Dict[str, Type[BaseForecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "ses": lambda **kw: ExponentialSmoothingForecaster(model_type="ses", **kw),
    "holt_linear": lambda **kw: ExponentialSmoothingForecaster(model_type="holt", **kw),
    "holt_winters": lambda **kw: ExponentialSmoothingForecaster(model_type="holt_winters", **kw),
    "arima": lambda **kw: ARIMAForecaster(is_seasonal=False, **kw),
    "sarima": lambda **kw: ARIMAForecaster(is_seasonal=True, **kw),
    "lightgbm": LightGBMForecaster,
    "random_forest": RandomForestForecaster,
    "xgboost": XGBoostForecaster,
}


def get_model(name: str, **kwargs: Any) -> BaseForecaster:
    """Instantiate a forecaster by identifier."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    factory = MODEL_REGISTRY[name]
    return factory(**kwargs)


def list_available_models() -> List[str]:
    """Return all registered model keys."""
    return list(MODEL_REGISTRY.keys())
