"""Phase 8 — Pydantic v2 request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


# -- Requests ----------------------------------------------------------------

class TrainRequest(BaseModel):
    states: list[str] = ["all"]
    n_jobs: int = 1


# -- Response building blocks ------------------------------------------------

class ForecastPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class PredictResponse(BaseModel):
    state: str
    forecast: list[ForecastPoint]
    selected_models: list[str]
    ensemble_weights: dict[str, float]


class BreakdownModel(BaseModel):
    model: str
    forecast: list[dict]


class BreakdownResponse(BaseModel):
    state: str
    models: dict[str, list[dict]]
    ensemble_weights: dict[str, float]


class BacktestRow(BaseModel):
    date: str
    y_true: float
    arima: float | None = None
    prophet: float | None = None
    xgboost: float | None = None
    lstm: float | None = None


class BacktestResponse(BaseModel):
    state: str
    rows: list[BacktestRow]


class TrainStatus(BaseModel):
    job_id: str
    status: str
    version: str | None = None
    error: str | None = None
