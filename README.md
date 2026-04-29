# Sales Forecasting System

End-to-end weekly sales forecasting for **43 US states** — Beverages category, 2019–2023.
Forecast next **8 weeks** per state using 4 ML models with automatic selection and ensemble.

## Stack

| Layer | Technology |
|---|---|
| **Models** | ARIMA/SARIMA (`pmdarima`, auto-order), Facebook Prophet (US holidays), XGBoost (direct multi-step, Optuna HPO, SHAP), LSTM (PyTorch CUDA 12.1, lookback=52) |
| **Service** | FastAPI + Pydantic v2 |
| **Dashboard** | React 18 + Recharts (Vite dev server) |
| **Registry** | Versioned artifacts under `artifacts/registry/vYYYYMMDD_HHMMSS/` |
| **Python** | 3.11 |

## Architecture

```
data/raw/sales.xlsx
        |
        v
  DataLoader  (mixed date formats: serial + DD-MM-YYYY)
        |
        v
  Preprocessor  (resample W-SUN → linear impute → IQR cap x3.0)
        |
        v
  walk_forward_splits  (5 folds, expanding window, no leakage)
        |
   +----+----+----------+----------+
   v         v          v          v
 ARIMA    Prophet   XGBoost     LSTM
(pmdarima) (US hols) (Optuna     (CUDA,
 D=0 fix)           50 trials)  lookback=52)
   |         |          |          |
   +----+----+----------+----------+
        |  rank by mean CV RMSE
        v
   top-2 ensemble  (inverse-RMSE weights)
        |
        v
   refit on full history → artifacts/registry/vTS/states/<State>/
        |
        v
   FastAPI (8 endpoints) <--> React Dashboard (3 tabs)
```

## Dataset

- **File:** place at `data/raw/sales.xlsx`
- **Rows:** 8 084 (43 US states x ~188 irregular dates)
- **Category:** Beverages
- **Date range:** 2019-01-12 to 2023-12-03
- **Target:** `Total` weekly sales

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
```

> **GPU requirement:** NVIDIA driver >= 525.60 for CUDA 12.1. Verify: `nvidia-smi`
> LSTM falls back to CPU automatically if CUDA is unavailable.

## Commands

| Task | Command |
|---|---|
| Train all 43 states | `python -m src.training.cli --states all --n-jobs 2` |
| Train one state | `python -m src.training.cli --states California --n-jobs 1` |
| Serve API | `uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload` |
| React dashboard | `cd dashboard/react-app && npm install && npm run dev` |
| Run tests | `pytest tests/ -v` |

> **Training time:** ~36 min/state (single-threaded). All 43 states with `--n-jobs 2` takes ~6-7 hrs.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{status, version, trained_states}` |
| `GET` | `/states` | `{trained: ["California", ...]}` |
| `POST` | `/train` | Start background training job → `{job_id, status}` |
| `GET` | `/train/{job_id}` | Poll job status → `{status, version?, error?}` |
| `GET` | `/predict?state=X&horizon=8` | 8-week ensemble forecast with widening CI bands |
| `GET` | `/predict/breakdown?state=X` | Per-model forecasts + ensemble weights |
| `GET` | `/metrics?state=X` | RMSE / MAE / MAPE / SMAPE per model across 5 CV folds |
| `GET` | `/backtest?state=X` | 40 rows (5 folds x 8 weeks): actual vs all 4 model predictions |

### Predict response shape

```json
{
  "state": "California",
  "forecast": [
    {"date": "2023-12-10", "yhat": 1276003121, "yhat_lower": 798090710, "yhat_upper": 1753915531},
    ...
  ],
  "selected_models": ["xgboost", "arima"],
  "ensemble_weights": {"xgboost": 0.5007, "arima": 0.4993}
}
```

> CI bands widen with horizon: `width = 1.96 * weighted_rmse * sqrt(h)` where h=1..8.

## Project Layout

```
.
+-- config.yaml                        # single source of truth (all hyperparams)
+-- requirements.txt                   # PyPI deps (numpy<2, prophet, pmdarima ...)
+-- requirements-torch.txt             # torch 2.5.1 + CUDA 12.1
+-- data/raw/sales.xlsx                # source dataset (place here, gitignored)
+-- artifacts/registry/v<TS>/          # versioned model bundles (gitignored)
|     +-- manifest.json
|     +-- states/<State>/
|           +-- arima.joblib / xgboost.joblib / lstm.joblib / prophet.joblib
|           +-- ensemble_weights.json
|           +-- cv_metrics.json         # per-fold scores + fold_details for backtest
|           +-- metadata.json
+-- src/
|   +-- data/          loader.py, preprocessor.py, splits.py
|   +-- features/      engineer.py  (20 features: lags, rolling, Fourier, holidays)
|   +-- models/        arima_model.py, prophet_model.py, xgboost_model.py, lstm_model.py
|   +-- training/      pipeline.py, cli.py
|   +-- evaluation/    metrics.py  (rmse, mae, mape, smape)
|   +-- api/           app.py, schemas.py
|   +-- utils/         registry.py, logging.py, seed.py
+-- dashboard/
|   +-- react-app/     Vite + React 18 + Recharts
|         +-- src/App.jsx, api.js, tabs/ForecastTab.jsx, BacktestTab.jsx, MetricsTab.jsx
+-- tests/
      +-- test_smoke.py
```

## Status

- [x] Phase 0 — Setup + scaffold
- [x] Phase 1 — Data pipeline (W-SUN resample, IQR cap x3, linear imputation)
- [x] Phase 2 — Feature engineering (20 features: lags, rolling, Fourier, holidays)
- [x] Phase 3 — ARIMA/SARIMA (auto-order, D=0 fix, non-seasonal fallback)
- [x] Phase 4 — Prophet (US holidays, yearly + weekly seasonality)
- [x] Phase 5 — XGBoost (8 direct models, Optuna HPO 50 trials, SHAP)
- [x] Phase 6 — LSTM (PyTorch CUDA, lookback=52, early stopping, state_dict save)
- [x] Phase 7 — Training pipeline (walk-forward CV, top-2 ensemble, registry)
- [x] Phase 8 — FastAPI service (8 endpoints, lazy loading, background training jobs)
- [x] Phase 9 — React dashboard (Forecast + CI, Backtest, Metrics tabs)
- [x] Phase 10 — Docs + video

## Edge Cases Handled

| Case | Handling |
|---|---|
| Irregular date gaps (1-91 days) | `resample('W-SUN').sum(min_count=1)` then linear interpolation |
| States with insufficient data | `walk_forward_splits` ValueError caught → state skipped with warning |
| LSTM on short series | Guard raises clear `ValueError` if `len(series) < lookback + horizon + batch_size` |
| XGBoost Optuna bad params | `subsample` / `colsample_bytree` clipped to [0.3, 1.0] after HPO |
| XGBoost fold-0 overfitting | Warning logged if fold-0 RMSE > 2x mean of other folds |
| Missing state in API | HTTP 404 with list of trained states |
| CI fixed-width | CI widens with `sqrt(h)` — week 8 has wider band than week 1 |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `"No trained models in registry"` | Run training first: `python -m src.training.cli --states California` |
| CUDA not available | LSTM falls back to CPU automatically — no action needed |
| Prophet install slow (first run) | Normal — Stan models compile once, ~5 min |
| ARIMA takes long | Normal — ~3 min/fold for 256 weeks |
| React dashboard blank | Ensure API is running on port 8000 before opening dashboard |
| `npm: command not found` | Install Node.js 18+ from https://nodejs.org |
