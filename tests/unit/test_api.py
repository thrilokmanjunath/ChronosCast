"""Unit tests for FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoints():
    res1 = client.get("/health")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "healthy"
    assert "version" in data1

    res2 = client.get("/api/v1/health")
    assert res2.status_code == 200
    assert res2.json()["status"] == "healthy"


def test_datasets_summary():
    res = client.get("/api/v1/datasets/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["store_count"] == 10
    assert data["item_count"] == 50
    assert data["total_observations"] == 913_000


def test_stores_endpoint():
    res = client.get("/api/v1/stores")
    assert res.status_code == 200
    stores = res.json()
    assert len(stores) == 10
    assert "total_sales" in stores[0]
    assert "sales_share_pct" in stores[0]


def test_items_endpoint():
    res = client.get("/api/v1/items")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 50
    assert "total_sales" in items[0]


def test_models_endpoint():
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    models = res.json()
    model_names = [m["model_name"] for m in models]
    assert "lightgbm" in model_names
    assert "seasonal_naive" in model_names


def test_forecast_endpoint_valid():
    payload = {
        "store": 1,
        "item": 1,
        "horizon": 7,
        "confidence_level": 0.90,
        "model": "lightgbm",
    }
    res = client.post("/api/v1/forecast", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["dates"]) == 7
    assert len(data["predictions"]) == 7
    assert len(data["lower_bounds"]) == 7
    assert len(data["upper_bounds"]) == 7
    assert data["store"] == 1
    assert data["item"] == 1
    # Verify bounds validity
    for p, lower_b, upper_b in zip(data["predictions"], data["lower_bounds"], data["upper_bounds"], strict=False):
        assert lower_b <= p <= upper_b


def test_forecast_endpoint_invalid_horizon():
    payload = {
        "store": 1,
        "item": 1,
        "horizon": 10,  # invalid horizon
        "confidence_level": 0.90,
    }
    res = client.post("/api/v1/forecast", json=payload)
    assert res.status_code == 400


def test_forecast_endpoint_invalid_store():
    payload = {
        "store": 99,  # invalid store id
        "item": 1,
        "horizon": 7,
        "confidence_level": 0.90,
    }
    res = client.post("/api/v1/forecast", json=payload)
    assert res.status_code == 422  # Pydantic validation failure


def test_model_performance():
    res = client.get("/api/v1/model/performance")
    assert res.status_code == 200
    data = res.json()
    assert "leaderboard" in data
    assert "test_report" in data
    assert "error_breakdown" in data
    assert len(data["leaderboard"]) > 0
