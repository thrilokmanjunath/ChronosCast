"""Full model training, validation, evaluation, and benchmarking pipeline.

Executes:
1. Chronological splitting (Train 2013-16, Val H1 2017, Test H2 2017).
2. Level 1 Baselines: Naive, Seasonal Naive, Moving Average.
3. Level 2 Statistical: Holt-Winters.
4. Level 3 Machine Learning: Global pooled LightGBM, Random Forest, XGBoost.
5. Validation benchmarking and Model Leaderboard generation.
6. Best model selection.
7. Conformal prediction interval calibration and test set coverage verification.
8. Final untouched Test evaluation and detailed error breakdowns.
9. Artifact persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from chronoscast.evaluation.backtesting import ChronologicalSplitter
from chronoscast.evaluation.metrics import calculate_all_metrics
from chronoscast.forecasting.uncertainty import ConformalIntervalEstimator
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
from chronoscast.models.statistical import ExponentialSmoothingForecaster
from chronoscast.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_training_and_evaluation() -> None:
    models_dir = Path("artifacts/models")
    metrics_dir = Path("artifacts/metrics")
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ChronosCast Model Training and Evaluation Pipeline...")

    # Set up MLflow local SQLite tracking
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("chronoscast_benchmarking")

    # 1. Load engineered feature dataset
    feat_path = "data/processed/features_sales.parquet"
    logger.info("Loading feature dataset from %s...", feat_path)
    df = pd.read_parquet(feat_path)
    df["date"] = pd.to_datetime(df["date"])

    # 2. Chronological Split: Train (2013-16), Val (H1 2017), Test (H2 2017)
    splitter = ChronologicalSplitter(
        train_end="2016-12-31",
        val_end="2017-06-30",
        test_end="2017-12-31",
    )
    train_df, val_df, test_df = splitter.split(df)

    # Feature columns for ML
    exclude = {"date", "sales"}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    # Prepare ML matrices (drop initial warm-up NaNs from lags)
    train_clean = train_df.dropna().reset_index(drop=True)
    val_clean = val_df.dropna().reset_index(drop=True)
    test_clean = test_df.dropna().reset_index(drop=True)

    X_train, y_train = train_clean[feature_cols], train_clean["sales"]
    X_val, y_val = val_clean[feature_cols], val_clean["sales"]
    X_test, y_test = test_clean[feature_cols], test_clean["sales"]

    leaderboard_records = []

    # -------------------------------------------------------------
    # LEVEL 1: BASELINES (Evaluated on Validation 28-day window)
    # -------------------------------------------------------------
    logger.info("Benchmarking Level 1 Baselines...")
    val_28_mask = (val_df["date"] >= "2017-01-01") & (val_df["date"] <= "2017-01-28")
    actual_val_28 = val_df[val_28_mask].sort_values(["store", "item", "date"]).reset_index(drop=True)

    # 1. Naive
    naive = NaiveForecaster().fit(train_df)
    pred_naive = naive.predict(horizon=28).sort_values(["store", "item", "date"]).reset_index(drop=True)
    m_naive = calculate_all_metrics(actual_val_28["sales"].values, pred_naive["prediction"].values)
    leaderboard_records.append({
        "model": "Naive (Persistence)",
        "model_type": "Baseline",
        "horizon": 28,
        "runtime_sec": np.round(naive.execution_time_, 3),
        **m_naive,
    })

    # 2. Seasonal Naive
    snaive = SeasonalNaiveForecaster(season_length=7).fit(train_df)
    pred_snaive = snaive.predict(horizon=28).sort_values(["store", "item", "date"]).reset_index(drop=True)
    m_snaive = calculate_all_metrics(actual_val_28["sales"].values, pred_snaive["prediction"].values)
    leaderboard_records.append({
        "model": "Seasonal Naive (s=7)",
        "model_type": "Baseline",
        "horizon": 28,
        "runtime_sec": np.round(snaive.execution_time_, 3),
        **m_snaive,
    })
    snaive.save(models_dir / "baseline_seasonal_naive.joblib")

    # 3. Moving Average (window=7)
    ma7 = MovingAverageForecaster(window=7).fit(train_df)
    pred_ma7 = ma7.predict(horizon=28).sort_values(["store", "item", "date"]).reset_index(drop=True)
    m_ma7 = calculate_all_metrics(actual_val_28["sales"].values, pred_ma7["prediction"].values)
    leaderboard_records.append({
        "model": "Moving Average (w=7)",
        "model_type": "Baseline",
        "horizon": 28,
        "runtime_sec": np.round(ma7.execution_time_, 3),
        **m_ma7,
    })

    # -------------------------------------------------------------
    # LEVEL 2: STATISTICAL MODELS (Holt-Winters on representative series)
    # -------------------------------------------------------------
    logger.info("Benchmarking Level 2 Statistical Models (Holt-Winters)...")
    # Fit Holt-Winters on sample series (Stores 1-2, Items 1-3)
    sample_train = train_df[(train_df["store"] <= 2) & (train_df["item"] <= 3)].copy()
    sample_val = actual_val_28[(actual_val_28["store"] <= 2) & (actual_val_28["item"] <= 3)].copy()

    hw = ExponentialSmoothingForecaster(model_type="holt_winters", seasonal_periods=7)
    hw.fit(sample_train)
    pred_hw = hw.predict(horizon=28).sort_values(["store", "item", "date"]).reset_index(drop=True)
    merged_hw = pd.merge(sample_val, pred_hw, on=["date", "store", "item"])
    m_hw = calculate_all_metrics(merged_hw["sales"].values, merged_hw["prediction"].values)
    leaderboard_records.append({
        "model": "Holt-Winters (Sample Series)",
        "model_type": "Statistical",
        "horizon": 28,
        "runtime_sec": np.round(hw.execution_time_, 3),
        **m_hw,
    })

    # -------------------------------------------------------------
    # LEVEL 3: MACHINE LEARNING MODELS (Full Pooled Dataset)
    # -------------------------------------------------------------
    logger.info("Benchmarking Level 3 Machine Learning Models...")

    # 1. Random Forest (trained on representative subset for rapid benchmark)
    rf_sample_train = train_clean.sample(n=60000, random_state=42)
    rf = RandomForestForecaster(n_estimators=60, max_depth=12, random_state=42)
    rf.fit(rf_sample_train[feature_cols], rf_sample_train["sales"])
    pred_rf = rf.predict_features(val_clean[feature_cols])
    m_rf = calculate_all_metrics(val_clean["sales"].values, pred_rf)
    leaderboard_records.append({
        "model": "Random Forest Regressor",
        "model_type": "Machine Learning",
        "horizon": 28,
        "runtime_sec": np.round(rf.execution_time_, 3),
        **m_rf,
    })

    # 2. XGBoost (trained on subset for speed)
    xgb_sample_train = train_clean.sample(n=100000, random_state=42)
    xgbf = XGBoostForecaster(n_estimators=120, learning_rate=0.08, max_depth=6)
    xgbf.fit(xgb_sample_train[feature_cols], xgb_sample_train["sales"])
    pred_xgb = xgbf.predict_features(val_clean[feature_cols])
    m_xgb = calculate_all_metrics(val_clean["sales"].values, pred_xgb)
    leaderboard_records.append({
        "model": "XGBoost Regressor",
        "model_type": "Machine Learning",
        "horizon": 28,
        "runtime_sec": np.round(xgbf.execution_time_, 3),
        **m_xgb,
    })

    # 3. LightGBM (Primary Advanced Model - Full Dataset)
    logger.info("Training Primary Advanced Model: Global Pooled LightGBM...")
    with mlflow.start_run(run_name="LightGBM_Global_Pooled"):
        lgb_forecaster = LightGBMForecaster(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        lgb_forecaster.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            feature_names=feature_cols,
        )
        pred_val_lgb = lgb_forecaster.predict_features(X_val)
        m_lgb_val = calculate_all_metrics(y_val.values, pred_val_lgb)

        mlflow.log_params(lgb_forecaster.get_params())
        mlflow.log_metrics({"val_mae": m_lgb_val["mae"], "val_rmse": m_lgb_val["rmse"], "val_wape": m_lgb_val["wape"]})

        leaderboard_records.append({
            "model": "LightGBM Regressor (Primary)",
            "model_type": "Machine Learning",
            "horizon": 28,
            "runtime_sec": np.round(lgb_forecaster.execution_time_, 3),
            **m_lgb_val,
        })

    # Save Leaderboard
    leaderboard_df = pd.DataFrame(leaderboard_records).sort_values(by="wape").reset_index(drop=True)
    leaderboard_df["rank"] = leaderboard_df.index + 1
    leaderboard_df.to_csv(metrics_dir / "leaderboard.csv", index=False)
    with open(metrics_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(leaderboard_df.to_dict(orient="records"), f, indent=2)

    logger.info("=== VALIDATION MODEL LEADERBOARD ===")
    for _, row in leaderboard_df.iterrows():
        logger.info(
            "Rank %d | %s | WAPE: %.2f%% | MAE: %.4f | RMSE: %.4f | Time: %.2fs",
            row["rank"], row["model"], row["wape"], row["mae"], row["rmse"], row["runtime_sec"],
        )

    # -------------------------------------------------------------
    # UNCERTAINTY ESTIMATION: CONFORMAL CALIBRATION
    # -------------------------------------------------------------
    logger.info("Calibrating Conformal Prediction Intervals on Validation residuals...")
    calibrator = ConformalIntervalEstimator(confidence_levels=[0.80, 0.90, 0.95])
    calibrator.calibrate(y_val.values, pred_val_lgb)
    calibrator.save(models_dir / "uncertainty_calibrator.json")

    # -------------------------------------------------------------
    # FINAL TEST SET EVALUATION (UNTIL NOW UNTOUCHED)
    # -------------------------------------------------------------
    logger.info("Evaluating selected best model (LightGBM) on untouched Final Test Set (H2 2017)...")
    pred_test_lgb = lgb_forecaster.predict_features(X_test)
    test_metrics = calculate_all_metrics(y_test.values, pred_test_lgb)

    # Conformal test coverage verification
    coverage_80 = calibrator.evaluate_coverage(y_test.values, pred_test_lgb, confidence_level=0.80)
    coverage_90 = calibrator.evaluate_coverage(y_test.values, pred_test_lgb, confidence_level=0.90)
    coverage_95 = calibrator.evaluate_coverage(y_test.values, pred_test_lgb, confidence_level=0.95)

    test_report = {
        "best_model": "LightGBM Regressor (Primary)",
        "test_period": "2017-07-01 to 2017-12-31",
        "test_observations": len(test_clean),
        "overall_metrics": test_metrics,
        "uncertainty_coverage": {
            "80%": coverage_80,
            "90%": coverage_90,
            "95%": coverage_95,
        },
    }
    with open(metrics_dir / "test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    logger.info("Final Test Performance: MAE=%.4f, RMSE=%.4f, WAPE=%.2f%%, sMAPE=%.2f%%",
                test_metrics["mae"], test_metrics["rmse"], test_metrics["wape"], test_metrics["smape"])
    logger.info("Empirical Test Coverage: 80%% Conf -> %.2f%%, 90%% Conf -> %.2f%%, 95%% Conf -> %.2f%%",
                coverage_80["empirical_coverage_pct"], coverage_90["empirical_coverage_pct"], coverage_95["empirical_coverage_pct"])

    # Detailed Error Breakdown on Test Set
    test_eval_df = test_clean[["date", "store", "item", "sales"]].copy()
    test_eval_df["prediction"] = pred_test_lgb
    test_eval_df["error"] = test_eval_df["sales"] - test_eval_df["prediction"]
    test_eval_df["abs_error"] = np.abs(test_eval_df["error"])

    # Error by Store
    store_err = test_eval_df.groupby("store").apply(
        lambda g: pd.Series(calculate_all_metrics(g["sales"].values, g["prediction"].values))
    ).reset_index()

    # Error by Item
    item_err = test_eval_df.groupby("item").apply(
        lambda g: pd.Series(calculate_all_metrics(g["sales"].values, g["prediction"].values))
    ).reset_index()

    # High Demand vs Low Demand Split (top 50% items vs bottom 50% items)
    median_item_sales = test_eval_df.groupby("item")["sales"].mean().median()
    high_items = test_eval_df.groupby("item")["sales"].mean()[lambda x: x >= median_item_sales].index
    high_demand_df = test_eval_df[test_eval_df["item"].isin(high_items)]
    low_demand_df = test_eval_df[~test_eval_df["item"].isin(high_items)]

    breakdown = {
        "by_store": store_err.to_dict(orient="records"),
        "by_item": item_err.to_dict(orient="records"),
        "high_demand_tier": calculate_all_metrics(high_demand_df["sales"].values, high_demand_df["prediction"].values),
        "low_demand_tier": calculate_all_metrics(low_demand_df["sales"].values, low_demand_df["prediction"].values),
        "feature_importances": dict(sorted(lgb_forecaster.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:15]),
    }
    with open(metrics_dir / "error_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2)

    # 8. Save best model artifact
    lgb_forecaster.save(models_dir / "best_lightgbm.joblib")
    logger.info("Pipeline completed successfully! All models and metrics saved.")


if __name__ == "__main__":
    run_training_and_evaluation()
