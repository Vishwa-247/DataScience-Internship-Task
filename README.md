# Sales Forecasting System

End-to-end weekly sales forecasting for **43 US states** (Beverages category, 2019–2023) — built per the [60-hour execution plan](60hr_execution_plan.html).

## Stack

- **Python** 3.11
- **Models:** ARIMA/SARIMA (`pmdarima`), Prophet, XGBoost (with Optuna HPO + SHAP), LSTM (PyTorch CUDA 12.1)
- **Service:** FastAPI + Pydantic v2
- **Dashboard:** Streamlit + Plotly

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
```

> **Driver requirement:** NVIDIA driver `>= 525.60` for CUDA 12.1 wheels. Verify with `nvidia-smi`.

Verify the install:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('cuda ok:', torch.cuda.is_available())"
```

## Project layout

```
.
├── config.yaml                   # single source of truth
├── requirements.txt              # PyPI deps (numpy<2, prophet, pmdarima, ...)
├── requirements-torch.txt        # torch CUDA 12.1
├── data/raw/sales.xlsx           # source dataset (gitignored)
├── artifacts/registry/v<TS>/     # versioned model bundles per state
├── src/
│   ├── data/         # loader, preprocessor, walk-forward splits
│   ├── features/     # FeatureEngineer (lags, rolling, Fourier, holidays)
│   ├── models/       # arima, prophet, xgboost, lstm
│   ├── training/     # pipeline + CLI
│   ├── evaluation/   # metrics
│   ├── api/          # FastAPI app + schemas
│   └── utils/        # seed, logging, registry
├── dashboard/app.py              # Streamlit
└── tests/
```

## Commands (will be populated as phases land)

| Phase | Command |
|-------|---------|
| Train all states | `python -m src.training.cli --states all` |
| Train one state | `python -m src.training.cli --states California` |
| Serve API | `uvicorn src.api.app:app --port 8000 --reload` |
| Dashboard | `streamlit run dashboard/app.py` |

## Status

- [x] Phase 0 — Setup + scaffold
- [ ] Phase 1 — Data pipeline
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — ARIMA/SARIMA
- [ ] Phase 4 — Prophet
- [ ] Phase 5 — XGBoost
- [ ] Phase 6 — LSTM
- [ ] Phase 7 — Training pipeline + auto-selection
- [ ] Phase 8 — FastAPI service
- [ ] Phase 9 — Streamlit dashboard
- [ ] Phase 10 — Docs + video
