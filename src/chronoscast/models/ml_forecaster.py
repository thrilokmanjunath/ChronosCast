"""Machine Learning forecasting models for ChronosCast.

Implements:
- LightGBMForecaster (Primary advanced global pooled model with direct horizons)
- RandomForestForecaster
- XGBoostForecaster
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

from chronoscast.models.base import BaseForecaster
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


class LightGBMForecaster(BaseForecaster):
    """Global pooled LightGBM model with direct multi-horizon capabilities."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="lightgbm",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs,
        )
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[tuple] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LightGBMForecaster:
        """Fit global LightGBM regressor on feature matrix X and target y."""
        t0 = time.time()
        self.feature_names_ = feature_names if feature_names else list(X_train.columns)

        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.params["n_estimators"],
            learning_rate=self.params["learning_rate"],
            num_leaves=self.params["num_leaves"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            random_state=self.params["random_state"],
            n_jobs=-1,
            verbose=-1,
        )

        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set if eval_set else None,
            callbacks=callbacks if callbacks else None,
        )

        # Feature importances
        raw_importances = self.model.feature_importances_
        total_imp = max(1.0, float(raw_importances.sum()))
        self.feature_importances_ = {
            col: float(np.round(imp / total_imp, 4))
            for col, imp in zip(self.feature_names_, raw_importances, strict=False)
        }

        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        logger.info("LightGBM fitted in %.2f seconds.", self.execution_time_)
        return self

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sales given an aligned feature matrix X."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def predict(self, horizon: int, X_inference: Optional[pd.DataFrame] = None, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts using inference features."""
        if X_inference is None:
            raise ValueError("LightGBM requires X_inference DataFrame with feature columns and metadata.")

        preds = self.predict_features(X_inference[self.feature_names_])
        result = X_inference[["date", "store", "item"]].copy()
        result["prediction"] = preds
        return result


class RandomForestForecaster(BaseForecaster):
    """Random Forest regressor baseline on engineered features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="random_forest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names_: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> RandomForestForecaster:
        t0 = time.time()
        self.feature_names_ = list(X_train.columns)
        # Convert categoricals to integers for sklearn
        X_num = X_train.copy()
        for col in X_num.select_dtypes(include=["category"]).columns:
            X_num[col] = X_num[col].cat.codes

        self.model.fit(X_num, y_train)
        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        logger.info("RandomForest fitted in %.2f seconds.", self.execution_time_)
        return self

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X_num = X.copy()
        for col in X_num.select_dtypes(include=["category"]).columns:
            X_num[col] = X_num[col].cat.codes
        preds = self.model.predict(X_num)
        return np.clip(preds, 0.0, None)

    def predict(self, horizon: int, X_inference: Optional[pd.DataFrame] = None, **kwargs: Any) -> pd.DataFrame:
        if X_inference is None:
            raise ValueError("RandomForest requires X_inference DataFrame.")
        preds = self.predict_features(X_inference[self.feature_names_])
        result = X_inference[["date", "store", "item"]].copy()
        result["prediction"] = preds
        return result


class XGBoostForecaster(BaseForecaster):
    """XGBoost regressor on engineered features."""

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="xgboost",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs,
        )
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            enable_categorical=True,
        )
        self.feature_names_: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> XGBoostForecaster:
        t0 = time.time()
        self.feature_names_ = list(X_train.columns)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        self.execution_time_ = time.time() - t0
        logger.info("XGBoost fitted in %.2f seconds.", self.execution_time_)
        return self

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, None)

    def predict(self, horizon: int, X_inference: Optional[pd.DataFrame] = None, **kwargs: Any) -> pd.DataFrame:
        if X_inference is None:
            raise ValueError("XGBoost requires X_inference DataFrame.")
        preds = self.predict_features(X_inference[self.feature_names_])
        result = X_inference[["date", "store", "item"]].copy()
        result["prediction"] = preds
        return result
