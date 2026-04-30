# Demo Guide — Sales Forecasting System

End-to-end reference for the live demo. All commands copy-paste ready.

---

## Pre-Demo Setup (do this 5 min before)

```powershell
# Terminal 1 — Start API
cd "D:\Data Science Internship"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Start React dashboard
cd "D:\Data Science Internship\dashboard\react-app"
npm run dev
```

Open in browser:
- API docs: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:5173

---

## Live Demo Script — 4 Minutes

### [0:00 – 0:20] The Problem

Open `data/raw/sales.xlsx` in Windows Explorer briefly.

> "This is 8,084 rows of weekly US beverage sales across 43 states from 2019 to 2023.
> The task: build a production-ready system that forecasts the next 8 weeks for any state,
> compares multiple models, auto-selects the best ones, and exposes everything via a REST API."

---

### [0:20 – 0:50] The Architecture

Show folder tree in VS Code / Windsurf sidebar.

> "The system has four layers:
> - Data pipeline: loader, preprocessor, walk-forward cross-validation
> - Four model families: ARIMA/SARIMA, Prophet, XGBoost with Optuna HPO, LSTM on CUDA
> - A training pipeline that auto-selects the top-2 models per state and builds an ensemble
> - FastAPI with 8 endpoints and a React dashboard with 3 tabs"

Point at `config.yaml`:
> "Every hyperparameter — IQR multiplier, CV folds, lookback window — lives in one config file."

---

### [0:50 – 1:20] Training Already Done

Show the terminal where training ran (or the log file).

> "I trained all 43 states using walk-forward cross-validation — 5 folds per state,
> 4 models per fold, all running in parallel. Total time: 3.4 hours using CUDA for LSTM."

Open the registry folder:
```
artifacts/registry/v20260429_214318/states/
```

> "Each state has its own folder: refitted models, ensemble weights, CV metrics, and
> fold-level predictions for backtesting — all versioned by timestamp."

---

### [1:20 – 2:00] API Live Demo — MOST IMPRESSIVE PART

Open http://127.0.0.1:8000/docs (Swagger UI)

**Step 1 — Health check**
```
GET /health
```
Expected:
```json
{"status": "ok", "version": "v20260429_214318", "trained_states": 43}
```
> "43 states trained, API is healthy."

**Step 2 — List states**
```
GET /states
```
> "All 43 US states are trained and ready."

**Step 3 — California prediction**
```
GET /predict?state=California
```
Point at response:
```json
{
  "selected_models": ["xgboost", "arima"],
  "ensemble_weights": {"xgboost": 0.5007, "arima": 0.4993},
  "forecast": [
    {"date": "...", "yhat": 1276003121, "yhat_lower": ..., "yhat_upper": ...},
    ...
  ]
}
```
> "XGBoost and ARIMA were the top-2 models for California. The CI bands widen with
> each horizon step — week 8 has a wider band than week 1. That's sqrt(h) scaling."

**Step 4 — Texas prediction (MONEY SHOT)**
```
GET /predict?state=Texas
```
Point at response:
```json
{
  "selected_models": ["arima", "prophet"],
  "ensemble_weights": {"arima": 0.51, "prophet": 0.49}
}
```
> "Texas selected ARIMA + Prophet — not XGBoost. The pipeline evaluates all 4 models
> independently for each state and picks the best two based on CV RMSE.
> This is real auto model selection — not hardcoded."

Also show Vermont:
```
GET /predict?state=Vermont
```
> "Vermont also picked Prophet over XGBoost. Small-state data patterns favour
> the trend/seasonality decomposition approach over gradient boosting."

---

### [2:00 – 3:00] React Dashboard

Open http://localhost:5173

**Forecast Tab:**
> "The KPI cards show Week 1, Week 4, and Week 8 forecasts. Notice the CI width
> card — it's larger for week 8 than week 1. That's the sqrt(h) widening."

Switch state: California → Texas
> "Watch the ensemble weights change — ARIMA/Prophet for Texas vs XGBoost/ARIMA
> for California. The dashboard is fully dynamic."

Scroll down to the breakdown table:
> "Every model's individual prediction side-by-side with the weighted ensemble."

**Backtest Tab:**
> "This is the walk-forward backtest. 5 CV folds × 8 weeks = 40 historical predictions.
> The vertical dashed lines mark fold boundaries. The green line is actual, the
> coloured dashed lines are model predictions. RMSE cards at the top show which
> model tracked best historically."

**Metrics Tab:**
> "Rank medals, IN ENSEMBLE badges for selected models, fold-by-fold RMSE trend
> chart showing consistency across the expanding training window."

Switch state to a small state (Vermont, Wyoming, Rhode Island):
> "Even small states with low volumes trained cleanly — the LSTM minimum-data
> guard and sparse-state skip logic prevented any crashes."

---

### [3:00 – 3:30] Code Quality

Show pytest:
```powershell
pytest tests/ -v
```
Expected: 6 green tests (7th needs Excel file).

> "Smoke tests cover config loading, no-leakage guarantee on CV splits,
> metric correctness, model interface contracts, and the LSTM short-series guard."

Briefly show `src/training/pipeline.py` line ~77:
> "Walk-forward splits are wrapped in a try/except — states with insufficient
> data are skipped with a clean warning, not a crash."

Briefly show `src/api/app.py` line ~194:
> "CI width: 1.96 × weighted RMSE × sqrt(i+1). Horizon-aware, statistically sound."

---

### [3:30 – 4:00] Wrap Up

Show GitHub repo: https://github.com/Vishwa-247/DataScience-Internship-Task

> "To summarise: 43 US states, 4 model families, automatic per-state model selection,
> inverse-RMSE ensemble with horizon-aware confidence intervals,
> 8 REST API endpoints, React dashboard with 3 analytical tabs.
> Walk-forward CV with strict no-leakage guarantees throughout.
> All 43 states trained in 3.4 hours on CUDA."

---

## Key Talking Points — Memorize These

| Point | What to say |
|---|---|
| **Auto model selection** | "Texas and Vermont chose Prophet over XGBoost — the system decided per-state, not hardcoded" |
| **CI widening** | "CI width = 1.96 × CV-RMSE × sqrt(h) — week 8 uncertainty is 2.8× week 1" |
| **No data leakage** | "Every test fold starts strictly after the last training date — assert enforced in code" |
| **Scale** | "43 states, 4 models, 5 folds = 860 model fits in 3.4 hours" |
| **IQR cap** | "Multiplier 3.0 — aggressive enough for outliers, preserves seasonal spikes" |
| **LSTM guard** | "Short-series ValueError raised before any cryptic PyTorch error" |
| **Optuna** | "50 trials of Bayesian HPO for XGBoost on fold 0, clipped to safe ranges" |

---

## If They Ask About Specific Files

| Question | Answer + file |
|---|---|
| "How does model selection work?" | `src/training/pipeline.py` lines 142–162: rank by mean RMSE, pick top-2 |
| "How is the ensemble weighted?" | `src/training/pipeline.py`: inverse-RMSE weights, normalised |
| "Where are CI bands computed?" | `src/api/app.py` lines 176–199 |
| "How is data preprocessed?" | `src/data/preprocessor.py`: W-SUN resample, linear impute, IQR cap |
| "How does LSTM work?" | `src/models/lstm_model.py`: lookback=52, 12 features, CUDA, early stop |
| "How does XGBoost HPO work?" | `src/models/xgboost_model.py`: Optuna TPE, 50 trials, params clipped |
| "How are features engineered?" | `src/features/engineer.py`: 20 features, fit on train only, no leakage |
| "Where are models stored?" | `artifacts/registry/v{timestamp}/states/{State}/` |

---

## API Quick Reference

```
GET  /health                        → {status, version, trained_states}
GET  /states                        → {trained: [43 state names]}
POST /train                         → {job_id} background job
GET  /train/{job_id}                → {status, version?, error?}
GET  /predict?state=X&horizon=8     → 8-week forecast + CI + ensemble_weights
GET  /predict/breakdown?state=X     → per-model forecasts + ensemble weights
GET  /metrics?state=X               → RMSE/MAE/MAPE/SMAPE per model (5 folds)
GET  /backtest?state=X              → 40 rows actual vs all 4 model predictions
```

---

## If Something Goes Wrong

| Problem | Fix |
|---|---|
| API returns 503 | Run `python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload` |
| Dashboard blank | Check API is running, check browser console for CORS errors |
| State not found | Run `GET /states` to confirm the state name exactly |
| React npm error | Run `cd dashboard/react-app` then `npm install` first |
| Port 8000 in use | `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess` |
