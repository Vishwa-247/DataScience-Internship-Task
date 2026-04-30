# Sales Forecasting System

> End-to-end weekly sales forecasting for **43 US states** — Beverages, 2019–2023.
> Forecasts the next **8 weeks** per state using 4 ML/DL models with automatic selection and inverse-RMSE ensemble.

---

## Table of Contents

- [Stack](#stack)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Setup](#setup)
- [Commands](#commands)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Status](#status)
- [Edge Cases Handled](#edge-cases-handled)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)

---

## Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Models** | ARIMA/SARIMA (`pmdarima`), Facebook Prophet (US holidays), XGBoost (Optuna HPO + SHAP), LSTM (PyTorch CUDA 12.1) |
| **API** | FastAPI + Pydantic v2 + Uvicorn |
| **Dashboard** | React 18 + Recharts + Vite |
| **Registry** | Versioned artifacts — `artifacts/registry/vYYYYMMDD_HHMMSS/` |

---

## Architecture

```
data/raw/sales.xlsx
        │
        ▼
  DataLoader        (mixed date formats: Excel serial + DD-MM-YYYY)
        │
        ▼
  Preprocessor      (resample W-SUN → linear impute → IQR cap ×3.0)
        │
        ▼
  walk_forward_splits  (5 folds, expanding window, strict no-leakage)
        │
   ┌────┴────┬──────────┬──────────┐
   ▼         ▼          ▼          ▼
 ARIMA    Prophet   XGBoost      LSTM
(D=0 fix) (US hols) (Optuna      (CUDA,
           )         50 trials)   lookback=52)
   │         │          │          │
   └────┬────┴──────────┴──────────┘
        │  rank by mean CV RMSE
        ▼
   top-2 ensemble  (inverse-RMSE weights)
        │
        ▼
   refit on full history → artifacts/registry/vTS/states/<State>/
        │
        ▼
   FastAPI (8 endpoints)  ◄──►  React Dashboard (3 tabs)
```

---

## Dataset

| Field | Value |
|---|---|
| **File** | Place at `data/raw/sales.xlsx` (gitignored) |
| **Rows** | 8,084 (43 US states × ~188 irregular dates) |
| **Category** | Beverages |
| **Date range** | 2019-01-12 → 2023-12-03 |
| **Target** | `Total` weekly sales |

---

## Setup

```powershell
# 1. Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 3. Install PyTorch with CUDA 12.1
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121

# 4. Install React dependencies
cd dashboard\react-app
npm install
cd ..\..
```

> **GPU:** NVIDIA driver ≥ 525.60 for CUDA 12.1. Verify with `nvidia-smi`.
> LSTM auto-falls back to CPU if CUDA is unavailable.

---

## Commands

| Task | Command |
|---|---|
| Train one state | `.\.venv\Scripts\python.exe -m src.training.cli --states California --n-jobs 1` |
| Train all 43 states | `.\.venv\Scripts\python.exe -m src.training.cli --states all --n-jobs 2` |
| Start API | `.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload` |
| Start dashboard | `cd dashboard\react-app` then `npm run dev` |
| Run tests | `.\.venv\Scripts\python.exe -m pytest tests/ -v` |

> **Training time:** ~36 min/state (single-threaded). All 43 states with `--n-jobs 2` ≈ 3.4 hours on CUDA.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status, registry version, staleness warning |
| `GET` | `/states` | List of all trained states |
| `POST` | `/train` | Start background training job → `{job_id}` |
| `GET` | `/train/{job_id}` | Poll job status |
| `GET` | `/predict?state=X&horizon=8` | 8-week ensemble forecast + widening CI bands |
| `GET` | `/predict/breakdown?state=X` | Per-model forecasts + ensemble weights |
| `GET` | `/metrics?state=X` | RMSE / MAE / MAPE / SMAPE per model across 5 CV folds |
| `GET` | `/backtest?state=X` | 40 rows (5 folds × 8 weeks): actual vs all 4 models |

**Swagger UI:** http://127.0.0.1:8000/docs

### Predict — Response Shape

```json
{
  "state": "California",
  "selected_models": ["xgboost", "arima"],
  "ensemble_weights": {"xgboost": 0.5007, "arima": 0.4993},
  "forecast": [
    {"date": "2023-12-10", "yhat": 1276003121, "yhat_lower": 798090710, "yhat_upper": 1753915531},
    "..."
  ]
}
```

> CI width = `1.96 × weighted_RMSE × √h` where h = 1…8. Week 8 band is 2.8× wider than week 1.

---

## Project Structure

```
.
├── config.yaml                    # Single source of truth (all hyperparams)
├── requirements.txt               # PyPI dependencies
├── requirements-torch.txt         # PyTorch 2.5.1 + CUDA 12.1
│
├── data/
│   └── raw/sales.xlsx             # Source dataset (gitignored — place here)
│
├── artifacts/
│   └── registry/v<timestamp>/     # Versioned model bundles (gitignored)
│       ├── manifest.json
│       └── states/<State>/
│           ├── arima.joblib
│           ├── xgboost.joblib
│           ├── lstm.joblib
│           ├── prophet.joblib
│           ├── ensemble_weights.json
│           ├── cv_metrics.json    # Per-fold scores + fold predictions
│           └── metadata.json
│
├── src/
│   ├── data/          # loader.py, preprocessor.py, splits.py
│   ├── features/      # engineer.py (20 features: lags, rolling, Fourier, holidays)
│   ├── models/        # arima_model.py, prophet_model.py, xgboost_model.py, lstm_model.py
│   ├── training/      # pipeline.py, cli.py
│   ├── evaluation/    # metrics.py (rmse, mae, mape, smape)
│   ├── api/           # app.py, schemas.py
│   └── utils/         # registry.py, logging.py, seed.py
│
├── dashboard/
│   └── react-app/     # Vite + React 18 + Recharts
│       └── src/       # App.jsx, api.js, tabs/ForecastTab.jsx, BacktestTab.jsx, MetricsTab.jsx
│
├── tests/
│   └── test_smoke.py  # 7 smoke tests (all passing)
│
└── docs/
    ├── DEMO_GUIDE.md  # Full 4-minute demo script with talking points
    └── CASE_STUDY.md  # Original assignment brief
```

---

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Setup + scaffold | ✅ |
| 1 | Data pipeline (W-SUN resample, IQR ×3, linear imputation) | ✅ |
| 2 | Feature engineering (20 features: lags, rolling, Fourier, holidays) | ✅ |
| 3 | ARIMA/SARIMA (auto-order, D=0 fix, non-seasonal fallback) | ✅ |
| 4 | Prophet (US holidays, yearly + weekly seasonality) | ✅ |
| 5 | XGBoost (8 direct models, Optuna HPO 50 trials, SHAP) | ✅ |
| 6 | LSTM (PyTorch CUDA, lookback=52, early stopping, state_dict) | ✅ |
| 7 | Training pipeline (walk-forward CV, top-2 ensemble, registry) | ✅ |
| 8 | FastAPI (8 endpoints, lazy loading, background jobs) | ✅ |
| 9 | React dashboard (Forecast + CI, Backtest, Metrics tabs) | ✅ |
| 10 | Docs + video | ✅ |

---

## Edge Cases Handled

| Case | Handling |
|---|---|
| Irregular date gaps (1–91 days) | `resample('W-SUN').sum(min_count=1)` then linear interpolation |
| States with insufficient data | `walk_forward_splits` ValueError caught → state skipped with warning |
| LSTM on short series | Guard raises `ValueError` before any cryptic PyTorch error |
| XGBoost Optuna bad params | `subsample` / `colsample_bytree` clipped to [0.3, 1.0] |
| XGBoost fold-0 overfitting | Warning logged if fold-0 RMSE > 2× mean of other folds |
| Missing state in API | HTTP 404 with list of trained states |
| Fixed-width CI | CI widens with `√h` — week 8 band is wider than week 1 |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `"No trained models in registry"` | Run training: `python -m src.training.cli --states California` |
| `WinError 10013` on API start | Use `--host 127.0.0.1` instead of `0.0.0.0` |
| CUDA not available | LSTM auto-falls back to CPU — no action needed |
| Prophet slow on first run | Normal — Stan compiles once (~5 min) |
| React dashboard blank | Ensure API is running on port 8000 first |
| `npm: command not found` | Install Node.js 18+ from https://nodejs.org |

---

## Documentation

- **Demo guide & talking points:** [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md)
- **Original assignment brief:** [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md)
- **API interactive docs:** http://127.0.0.1:8000/docs (when server is running)
