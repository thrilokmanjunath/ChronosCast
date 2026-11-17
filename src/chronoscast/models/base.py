"""Base abstract forecaster interface for ChronosCast."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseForecaster(ABC):
    """Abstract Base Class for all ChronosCast forecasting models."""

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.params = kwargs
        self.is_fitted = False
        self.execution_time_: float = 0.0

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, **kwargs: Any) -> BaseForecaster:
        """Fit the model to the training DataFrame."""
        pass

    @abstractmethod
    def predict(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts for the specified horizon (days).

        Returns:
            DataFrame with columns ['date', 'store', 'item', 'prediction'].
        """
        pass

    def save(self, filepath: str | Path) -> None:
        """Serialize model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved %s model to %s", self.name, path)

    @classmethod
    def load(cls, filepath: str | Path) -> BaseForecaster:
        """Deserialize model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        model = joblib.load(path)
        logger.info("Loaded %s model from %s", getattr(model, "name", cls.__name__), path)
        return model

    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters."""
        return self.params
