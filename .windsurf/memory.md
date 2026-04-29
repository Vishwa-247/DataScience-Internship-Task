# Project: Sales Forecasting System (60h Build)

## 1. DONE
- [x] Phase 0: Python 3.11 venv created (file: .venv/)
- [x] Phase 0: requirements files authored (files: requirements.txt, requirements-torch.txt)
- [x] Phase 0: Main stack installed (90 pkgs, numpy<2 pinned)
- [x] Phase 0: Torch CUDA 12.1 installed, GPU verified (file: requirements-torch.txt)
- [x] Phase 0: Folder skeleton created (folders: src/, dashboard/, tests/, data/raw/, artifacts/registry/)
- [x] Phase 0: config.yaml authored (file: config.yaml)
- [x] Phase 0: utils trio impl (files: src/utils/seed.py, logging.py, registry.py)
- [x] Phase 0: README stub + .gitignore (files: README.md, .gitignore)
- [x] Phase 1: loader.py — mixed dates (dayfirst=True), schema validate (file: src/data/loader.py)
- [x] Phase 1: preprocessor.py — W-SUN resample(min_count=1), interp, IQR cap (file: src/data/preprocessor.py)
- [x] Phase 1: splits.py — walk-forward CV, 5 folds, no leakage (file: src/data/splits.py)
- [x] Phase 1: Integration tested — 43 states × 256 weeks, 0 NaN
- [x] Phase 2: FeatureEngineer — fit/transform, 20 cols, warmup=30, no test NaN (file: src/features/engineer.py)
- [x] Phase 3: ARIMAModel — auto_arima D=0 fallback, uniform interface (file: src/models/arima_model.py)
- [x] Phase 4: ProphetModel — ds/y rename, US holidays, uniform interface (file: src/models/prophet_model.py)
- [x] Phase 5: XGBoostModel — 8 direct models, Optuna HPO, SHAP (file: src/models/xgboost_model.py)
- [x] Phase 6: LSTMModel — lookback=52, MinMaxScaler, CUDA, early stop (file: src/models/lstm_model.py)
- [x] Phase 7: metrics.py + pipeline.py + cli.py — CV, top-2, inverse-RMSE ensemble, registry (files: src/evaluation/metrics.py, src/training/pipeline.py, src/training/cli.py)
- [x] Phase 8: FastAPI — 8 endpoints, lazy registry, CORS, CI from RMSE (files: src/api/app.py, src/api/schemas.py)
- [x] Phase 9: React dashboard — Vite+Recharts, 3 tabs, sidebar state selector (dir: dashboard/react-app/)

## 2. TODO
- [ ] Phase 10: README full docs + 5-8 min video

## 3. CONTEXT
**What**: Weekly sales forecast for 43 US states, Beverages, 8-week horizon
**Stack**: Py3.11 venv, numpy<2, pandas2.2, prophet1.3, pmdarima2.1.1, xgboost3.2, torch2.5.1+cu121, fastapi, streamlit
**Entry**: src/training/cli.py (training), src/api/app.py (serving), dashboard/app.py (UI)
**Data**: data/raw/sales.xlsx (8084 rows, 43 states x ~188 dates, irregular gaps 1-91d, must resample W-SUN)
**GPU**: GTX 1650 cap 7.5, driver 555.97, CUDA 12.1 wheel
**Last**: Phase 8+9 FastAPI+React dashboard complete 2026-04-29
