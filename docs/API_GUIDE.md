# ChronosCast — REST API Reference Guide

The ChronosCast backend is implemented with FastAPI and conforms to OpenAPI 3.1 standards.
Interactive Swagger documentation is available at `http://localhost:8000/docs` and Redoc at `http://localhost:8000/redoc`.

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### 1. System Health Check
`GET /health` or `GET /api/v1/health`

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "app_name": "ChronosCast API",
  "version": "0.1.0",
  "timestamp": "2026-11-17T20:57:45.000000Z"
}
```

### 2. Dataset Descriptive Summary
`GET /api/v1/datasets/summary`

**Response (200 OK)**:
```json
{
  "total_sales": 47708520.0,
  "mean_total_daily_sales": 26127.34,
  "store_count": 10,
  "item_count": 50,
  "start_date": "2013-01-01",
  "end_date": "2017-12-31",
  "total_days": 1826,
  "total_observations": 913000
}
```

### 3. List Stores
`GET /api/v1/stores`

**Response (200 OK)**:
```json
[
  {
    "store": 2,
    "total_sales": 6120128.0,
    "mean_sales": 67.03,
    "std_sales": 33.88,
    "sales_share_pct": 12.83
  }
]
```

### 4. List Items
`GET /api/v1/items`

**Response (200 OK)**:
```json
[
  {
    "item": 15,
    "total_sales": 1699775.0,
    "mean_sales": 93.09,
    "std_sales": 31.30,
    "sales_share_pct": 3.56
  }
]
```

### 5. Generate Sales Forecast
`POST /api/v1/forecast`

**Request Body**:
```json
{
  "store": 1,
  "item": 1,
  "horizon": 14,
  "confidence_level": 0.90,
  "model": "lightgbm"
}
```

**Response (200 OK)**:
```json
{
  "status": "success",
  "model": "lightgbm",
  "aggregation_type": "individual_series",
  "series_title": "Store 1 — Item 1",
  "store": 1,
  "item": 1,
  "horizon": 14,
  "confidence_level": 0.9,
  "generated_at": "2026-11-17T21:00:00.000000Z",
  "dates": ["2018-01-01", "2018-01-02", "..."],
  "predictions": [16.82, 17.45, "..."],
  "lower_bounds": [4.26, 4.89, "..."],
  "upper_bounds": [29.38, 30.01, "..."],
  "forecast_total": 248.5
}
```

### 6. Model Benchmark & Leaderboard
`GET /api/v1/model/performance`

**Response (200 OK)**:
Returns leaderboard rankings, test set evaluation metrics, and slice error breakdowns.
