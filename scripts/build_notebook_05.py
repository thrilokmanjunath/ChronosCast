"""Generate 05_model_evaluation.ipynb."""

from pathlib import Path

import nbformat as nbf

notebooks_dir = Path("notebooks")
nb = nbf.v4.new_notebook()
nb.metadata = {
    "language_info": {"name": "python", "version": "3.12"},
    "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
}

cells = [
    nbf.v4.new_markdown_cell("""# ChronosCast — Notebook 05: Model Evaluation, Benchmarking & Uncertainty
**Project**: ChronosCast: Intelligent Time Series Trend Analysis & Forecasting Platform

### Evaluation Methodology:
1. **Chronological Holdout Validation**:
   - Train: 2013-01-01 to 2016-12-31 (730,500 observations)
   - Validation: 2017-01-01 to 2017-06-30 (90,500 observations)
   - Final Test: 2017-07-01 to 2017-12-31 (92,000 observations)
2. **Benchmark Leaderboard**: Baselines vs Statistical vs Machine Learning
3. **Residual Diagnostics & Calibration**: Non-parametric Conformal Prediction Intervals
4. **Final Test Verification**: Error analysis by Store, Item, and Demand Volume Tiers.
"""),
    nbf.v4.new_code_cell("""import json
import pandas as pd
import numpy as np

# Load benchmark leaderboard
leaderboard = pd.read_csv("../artifacts/metrics/leaderboard.csv")
leaderboard
"""),
    nbf.v4.new_markdown_cell("### 1. Final Test Set Evaluation Report"),
    nbf.v4.new_code_cell("""with open("../artifacts/metrics/test_evaluation_report.json") as f:
    test_report = json.load(f)

print(f"Best Selected Model: {test_report['best_model']}")
print(f"Test Period: {test_report['test_period']} ({test_report['test_observations']:,} observations)")
print("Test Accuracy Metrics:")
for k, v in test_report["overall_metrics"].items():
    print(f"  {k.upper()}: {v}")

print("\\nEmpirical Prediction Interval Coverage:")
for conf, stats in test_report["uncertainty_coverage"].items():
    print(f"  Nominal {conf} -> Empirical Coverage: {stats['empirical_coverage_pct']}% (Margin: +/-{stats['margin_of_error']} units)")
"""),
    nbf.v4.new_markdown_cell("### 2. Error Breakdown by Demand Tier & Feature Importance"),
    nbf.v4.new_code_cell("""with open("../artifacts/metrics/error_breakdown.json") as f:
    breakdown = json.load(f)

print("High Demand Tier:", breakdown["high_demand_tier"])
print("Low Demand Tier:", breakdown["low_demand_tier"])
print("\\nTop 10 Feature Importances:")
for feat, imp in list(breakdown["feature_importances"].items())[:10]:
    print(f"  {feat:20s}: {imp * 100:.2f}%")
""")
]

nb.cells = cells
with open(notebooks_dir / "05_model_evaluation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Created 05_model_evaluation.ipynb")
