# ChronosCast — Developer & Contributor Guide

## 1. Local Environment Setup

### Prerequisites
- Python 3.12
- `uv` (recommended) or standard `pip`

```bash
# Clone the repository
git clone https://github.com/thrilokmanjunath/ChronosCast.git
cd ChronosCast

# Create virtual environment with uv
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install all dependencies in editable mode
uv pip install -e ".[dev]"
```

## 2. Running Data Pipeline and Models
```bash
# 1. Download raw Kaggle dataset
python scripts/download_data.py

# 2. Clean, validate, and produce interim parquet
python src/chronoscast/data/cleaner.py

# 3. Generate exploratory figures and diagnostics
python scripts/generate_eda_artifacts.py

# 4. Engineer features across 500 store-item series
python scripts/build_features.py

# 5. Train all model tiers, benchmark leaderboard, and calibrate conformal intervals
python scripts/train_and_evaluate.py
```

## 3. Running Applications
```bash
# Start FastAPI backend (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit interactive dashboard (port 8501)
streamlit run app/Home.py --server.port 8501
```

## 4. Running Tests and Linting
```bash
# Run complete test suite (35 tests)
pytest tests/ -v

# Run test coverage
pytest tests/ --cov=chronoscast --cov-report=term-missing

# Run Ruff linting and formatting checks
ruff check src/ api/ app/ tests/ scripts/
ruff format --check src/ api/ app/ tests/ scripts/
```

## 5. Docker Execution
```bash
# Build and start both API and Dashboard
docker compose up --build

# Access:
# API Docs: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```
