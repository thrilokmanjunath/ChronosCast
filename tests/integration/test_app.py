"""Integration tests for application startup, data integrity, and forecast generation."""

from pathlib import Path

from chronoscast.forecasting.engine import ForecastingEngine


def test_data_artifacts_exist():
    assert Path("data/raw/train.csv").exists()
    assert Path("data/interim/cleaned_sales.parquet").exists()
    assert Path("data/processed/features_sales.parquet").exists()


def test_model_and_metric_artifacts_exist():
    assert Path("artifacts/models/best_lightgbm.joblib").exists()
    assert Path("artifacts/models/uncertainty_calibrator.json").exists()
    assert Path("artifacts/metrics/leaderboard.csv").exists()
    assert Path("artifacts/metrics/test_evaluation_report.json").exists()
    assert Path("artifacts/metrics/error_breakdown.json").exists()


def test_forecasting_engine_individual_series():
    engine = ForecastingEngine()
    res = engine.generate_forecast(store=1, item=1, horizon=7, confidence_level=0.90)
    assert res["status"] == "success"
    assert len(res["dates"]) == 7
    assert len(res["predictions"]) == 7
    assert res["aggregation_type"] == "individual_series"
    assert res["forecast_total"] > 0


def test_forecasting_engine_store_aggregate():
    engine = ForecastingEngine()
    res = engine.generate_forecast(store=2, item=None, horizon=14, confidence_level=0.95)
    assert res["status"] == "success"
    assert len(res["dates"]) == 14
    assert res["aggregation_type"] == "store_aggregated"
    # Store aggregated demand across 50 items should be higher than a single item
    assert res["forecast_total"] > 1000


def test_forecasting_engine_network_total():
    engine = ForecastingEngine()
    res = engine.generate_forecast(store=None, item=None, horizon=28, confidence_level=0.90)
    assert res["status"] == "success"
    assert len(res["dates"]) == 28
    assert res["aggregation_type"] == "network_total"
    assert res["forecast_total"] > 50000


def test_forecasting_engine_recursive_progression():
    engine = ForecastingEngine()
    res = engine.generate_forecast(store=1, item=1, horizon=14, confidence_level=0.90)
    preds = res["predictions"]
    # Verify non-trivial non-flat progression across days
    assert len(set(preds)) > 5
    # Verify bounds
    for p, l, u in zip(preds, res["lower_bounds"], res["upper_bounds"], strict=False):
        assert l <= p <= u


def test_app_pages_syntax_and_importability():
    # Verify Streamlit page scripts can be compiled and parsed without syntax errors
    pages = [
        "app/Home.py",
        "app/pages/1_Data_Explorer.py",
        "app/pages/2_Trend_Analysis.py",
        "app/pages/3_Forecast_Studio.py",
        "app/pages/4_Model_Comparison.py",
        "app/pages/5_Forecast_Evaluation.py",
        "app/pages/6_Inventory_Insights.py",
        "app/pages/7_Reports.py",
    ]
    for page in pages:
        with open(page, "r", encoding="utf-8") as f:
            code = f.read()
        compile(code, page, "exec")
