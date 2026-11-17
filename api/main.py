"""FastAPI entry point for ChronosCast platform."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from api.schemas import HealthResponse
from chronoscast.utils.logger import setup_logger

logger = setup_logger("chronoscast.api")

app = FastAPI(
    title="ChronosCast API",
    description="Intelligent Time Series Trend Analysis & Forecasting Platform API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Streamlit / external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 router
app.include_router(router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def root_health() -> HealthResponse:
    """Root health check endpoint."""
    return HealthResponse(
        status="healthy",
        app_name="ChronosCast API",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error processing %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )
