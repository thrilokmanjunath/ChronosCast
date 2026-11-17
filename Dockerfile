FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CHRONOSCAST_ENV=production

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for lightning fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project definition and install dependencies
COPY pyproject.toml README.md ./
RUN uv pip install --system -e ".[dev]"

# Copy project directories
COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/
COPY tests/ ./tests/

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default entrypoint starts API and Streamlit or via docker-compose
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0"]
