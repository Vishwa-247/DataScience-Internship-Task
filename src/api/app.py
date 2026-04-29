"""Phase 8 — FastAPI app.

Lazy-loads registry on startup; caches per-state model bundles on first
request.  ``POST /train`` triggers background training.

Run::

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BacktestResponse,
    BacktestRow,
    BreakdownResponse,
    ForecastPoint,
    PredictResponse,
    TrainRequest,
    TrainStatus,
)
from src.data.loader import load_config, load_sales
from src.data.preprocessor import Preprocessor
from src.training.pipeline import MODEL_CLASSES, ARTIFACT_NAMES, TrainingPipeline
from src.utils.logging import get_logger
from src.utils.registry import Registry

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Sales Forecasting API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------

_config: dict[str, Any] = {}
_registry: Registry | None = None
_state_cache: dict[str, dict] = {}          # state → {models, weights, cv, meta}
_train_jobs: dict[str, dict[str, Any]] = {} # job_id → {status, version?, error?}


@app.on_event("startup")
def _startup() -> None:
    global _config, _registry
    _config = load_config()
    _registry = Registry(_config["artifacts"]["registry_path"])
    logger.info("API started — registry at %s", _registry.root)


def _get_version() -> str:
    ver = _registry.latest_version()
    if ver is None:
        raise HTTPException(503, "No trained models in registry. Run training first.")
    return ver


def _load_state(state: str) -> dict:
    """Lazy-load a state bundle into the cache. Returns cached dict."""
    if state in _state_cache:
        return _state_cache[state]

    ver = _get_version()
    trained = _registry.list_states(ver)
    if state not in trained:
        raise HTTPException(
            404,
            f"State '{state}' not trained yet. Trained: {trained}",
        )

    state_dir = _registry.state_dir(ver, state)

    # Load ensemble weights
    wpath = state_dir / "ensemble_weights.json"
    weights = json.loads(wpath.read_text(encoding="utf-8")) if wpath.exists() else {}

    # Load metadata
    meta = _registry.read_metadata(ver, state)

    # Load cv_metrics
    cv_path = state_dir / "cv_metrics.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8")) if cv_path.exists() else {}

    # Load selected models lazily
    models: dict[str, Any] = {}
    for model_name in meta.get("selected_models", []):
        art = state_dir / ARTIFACT_NAMES.get(model_name, f"{model_name}.joblib")
        if art.exists():
            cls = MODEL_CLASSES[model_name]
            models[model_name] = cls.load(art)
            logger.info("Loaded %s for %s", model_name, state)

    bundle = {"models": models, "weights": weights, "cv": cv, "meta": meta}
    _state_cache[state] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    ver = _registry.latest_version() if _registry else None
    trained = _registry.list_states(ver) if ver else []
    return {"status": "ok", "version": ver, "trained_states": len(trained)}


@app.get("/states")
def states():
    ver = _get_version()
    return {"trained": _registry.list_states(ver)}


# ---- Training (background) ------------------------------------------------

@app.post("/train")
def train(req: TrainRequest, bg: BackgroundTasks):
    job_id = str(uuid4())[:8]
    _train_jobs[job_id] = {"status": "running"}
    bg.add_task(_run_training, job_id, req.states, req.n_jobs)
    return {"job_id": job_id, "status": "queued"}


def _run_training(job_id: str, states_req: list[str], n_jobs: int) -> None:
    try:
        df = load_sales(config=_config)
        state_series = Preprocessor(_config).transform(df)
        pipeline = TrainingPipeline(_config)
        s = None if states_req == ["all"] else states_req
        version, results = pipeline.run(state_series, states=s, n_jobs=n_jobs)
        _train_jobs[job_id] = {"status": "completed", "version": version}
        _state_cache.clear()  # invalidate cache
    except Exception as exc:
        logger.error("Training job %s failed: %s", job_id, exc)
        _train_jobs[job_id] = {"status": "failed", "error": str(exc)}


@app.get("/train/{job_id}", response_model=TrainStatus)
def train_status(job_id: str):
    if job_id not in _train_jobs:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    return TrainStatus(job_id=job_id, **_train_jobs[job_id])


# ---- Prediction -----------------------------------------------------------

@app.get("/predict", response_model=PredictResponse)
def predict(state: str = Query(...), horizon: int = Query(8, ge=1, le=52)):
    bundle = _load_state(state)
    weights = bundle["weights"]
    meta = bundle["meta"]
    selected = meta.get("selected_models", [])

    # CI: ±1.96 × weighted avg RMSE from CV
    cv = bundle["cv"]
    rmse_vals = [cv.get(m, {}).get("mean_rmse", 0) for m in selected]
    w_list = [weights.get(m, 0) for m in selected]
    w_sum = sum(w_list) or 1
    ci_std = sum(r * w for r, w in zip(rmse_vals, w_list)) / w_sum

    # Ensemble prediction: weighted average of selected models
    preds_by_model = {}
    for m_name, m_obj in bundle["models"].items():
        preds_by_model[m_name] = m_obj.predict(horizon=horizon)

    # Weighted sum
    first = list(preds_by_model.values())[0]
    ensemble = np.zeros(horizon)
    for m_name, pred in preds_by_model.items():
        ensemble += weights.get(m_name, 0) * pred.values[:horizon]

    forecast = []
    for i, dt in enumerate(first.index[:horizon]):
        forecast.append(ForecastPoint(
            date=str(dt.date()),
            yhat=round(float(ensemble[i]), 2),
            yhat_lower=round(float(ensemble[i] - 1.96 * ci_std), 2),
            yhat_upper=round(float(ensemble[i] + 1.96 * ci_std), 2),
        ))

    return PredictResponse(
        state=state,
        forecast=forecast,
        selected_models=selected,
        ensemble_weights={m: round(w, 4) for m, w in weights.items()},
    )


@app.get("/predict/breakdown", response_model=BreakdownResponse)
def predict_breakdown(state: str = Query(...), horizon: int = Query(8)):
    bundle = _load_state(state)
    models_out: dict[str, list[dict]] = {}
    for m_name, m_obj in bundle["models"].items():
        pred = m_obj.predict(horizon=horizon)
        models_out[m_name] = [
            {"date": str(d.date()), "yhat": round(float(v), 2)}
            for d, v in zip(pred.index, pred.values)
        ]
    return BreakdownResponse(
        state=state,
        models=models_out,
        ensemble_weights=bundle["weights"],
    )


# ---- Metrics ---------------------------------------------------------------

@app.get("/metrics")
def metrics(state: str = Query(None)):
    ver = _get_version()
    trained = _registry.list_states(ver)

    if state:
        if state not in trained:
            raise HTTPException(404, f"State '{state}' not trained yet.")
        bundle = _load_state(state)
        cv = bundle["cv"]
        return {
            "state": state,
            "models": {
                m: {"mean_rmse": cv[m]["mean_rmse"], "folds": cv[m]["folds"]}
                for m in ["arima", "prophet", "xgboost", "lstm"]
                if m in cv
            },
        }

    # All trained states — summary only
    out = {}
    for s in trained:
        b = _load_state(s)
        cv = b["cv"]
        out[s] = {
            m: round(cv[m]["mean_rmse"])
            for m in ["arima", "prophet", "xgboost", "lstm"]
            if m in cv
        }
    return {"states": out}


# ---- Backtest --------------------------------------------------------------

@app.get("/backtest", response_model=BacktestResponse)
def backtest(state: str = Query(...)):
    bundle = _load_state(state)
    cv = bundle["cv"]
    fold_details = cv.get("fold_details")

    if not fold_details:
        raise HTTPException(
            422,
            "Backtest data not available. Re-run training to populate fold predictions.",
        )

    rows: list[BacktestRow] = []
    for fd in fold_details:
        dates = fd["dates"]
        actuals = fd["actuals"]
        preds = fd["predictions"]
        for i, dt in enumerate(dates):
            rows.append(BacktestRow(
                date=dt,
                y_true=actuals[i],
                arima=preds.get("arima", [None] * len(dates))[i],
                prophet=preds.get("prophet", [None] * len(dates))[i],
                xgboost=preds.get("xgboost", [None] * len(dates))[i],
                lstm=preds.get("lstm", [None] * len(dates))[i],
            ))

    return BacktestResponse(state=state, rows=rows)
