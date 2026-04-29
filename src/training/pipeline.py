"""Phase 7 — Walk-forward CV across all model families per state, auto-select top-2,
inverse-RMSE ensemble, refit on full history, write registry bundle.

Per-state pipeline::

    walk-forward CV (5 folds)
    → score 4 models (ARIMA, Prophet, XGBoost, LSTM)
    → rank by mean RMSE
    → pick top-2
    → ensemble via inverse-RMSE weighted average
    → refit top-2 on full history
    → serialize to registry

Parallelised across states with ``joblib.Parallel(n_jobs=-1, backend='loky')``.
LSTM workers set ``torch.set_num_threads(1)`` to avoid CUDA contention.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.splits import walk_forward_splits
from src.evaluation.metrics import score_forecast
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.utils.logging import get_logger
from src.utils.registry import Registry

logger = get_logger(__name__)

MODEL_CLASSES: dict[str, type] = {
    "arima": ARIMAModel,
    "prophet": ProphetModel,
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
}

ARTIFACT_NAMES: dict[str, str] = {
    "arima": "arima.joblib",
    "prophet": "prophet.joblib",
    "xgboost": "xgboost.joblib",
    "lstm": "lstm.joblib",
}


# ---------------------------------------------------------------------------
# Single-state pipeline (runs inside a worker)
# ---------------------------------------------------------------------------

def _train_one_state(
    state: str,
    series: pd.Series,
    config: dict[str, Any],
    registry: Registry,
    version: str,
) -> dict[str, Any]:
    """Full pipeline for a single state. Returns summary dict.

    Called by ``joblib.Parallel`` — must be a top-level function (picklable).
    """
    # Avoid CUDA contention from parallel Loky workers
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")

    horizon = config["forecast"]["horizon_weeks"]
    ensemble_cfg = config["ensemble"]

    folds = list(walk_forward_splits(series, config))
    n_folds = len(folds)
    logger.info("[%s] Starting CV — %d folds, %d weeks", state, n_folds, len(series))

    # ------------------------------------------------------------------
    # 1. Walk-forward CV: score every model on every fold
    # ------------------------------------------------------------------
    # fold_scores[model_name] = [{"rmse": …, "mae": …, …}, …]  per fold
    fold_scores: dict[str, list[dict[str, float]]] = {m: [] for m in MODEL_CLASSES}
    fold_preds: dict[str, list[np.ndarray]] = {m: [] for m in MODEL_CLASSES}
    fold_actuals: list[np.ndarray] = []
    fold_dates: list[list[str]] = []

    # XGBoost: optimize on fold 0, reuse params
    xgb_best_params: dict | None = None

    for fold_idx, (train, test) in enumerate(folds):
        fold_actuals.append(test.values)
        fold_dates.append([str(d.date()) for d in test.index])

        for model_name, model_cls in MODEL_CLASSES.items():
            try:
                model = model_cls(config)

                # XGBoost HPO: fold 0 only
                if model_name == "xgboost":
                    if fold_idx == 0:
                        model.optimize(train, test, n_trials=config["models"]["xgboost"].get("optuna_trials", 50))
                        xgb_best_params = model.params.copy()
                    elif xgb_best_params is not None:
                        model.params.update(xgb_best_params)

                model.fit(train)
                pred = model.predict(horizon=horizon)

                scores = score_forecast(test.values, pred.values)
                fold_scores[model_name].append(scores)
                fold_preds[model_name].append(pred.values)

            except Exception as exc:
                logger.warning("[%s] %s fold %d failed: %s", state, model_name, fold_idx, exc)
                fold_scores[model_name].append({"rmse": float("inf"), "mae": float("inf"),
                                                 "mape": float("inf"), "smape": float("inf")})
                fold_preds[model_name].append(np.full(horizon, np.nan))

    # ------------------------------------------------------------------
    # 2. Rank by mean RMSE → pick top-K
    # ------------------------------------------------------------------
    mean_rmse: dict[str, float] = {}
    for model_name, scores_list in fold_scores.items():
        rmse_vals = [s["rmse"] for s in scores_list]
        mean_rmse[model_name] = float(np.mean(rmse_vals))

    ranked = sorted(mean_rmse, key=mean_rmse.get)
    top_k = ensemble_cfg.get("top_k", 2)
    selected = ranked[:top_k]

    logger.info(
        "[%s] Model ranking (mean RMSE): %s → selected %s",
        state,
        {m: f"{mean_rmse[m]:,.0f}" for m in ranked},
        selected,
    )

    # ------------------------------------------------------------------
    # 3. Ensemble weights: inverse-RMSE
    # ------------------------------------------------------------------
    if ensemble_cfg.get("weighting", "inverse_rmse") == "inverse_rmse":
        inv = {m: 1.0 / max(mean_rmse[m], 1e-10) for m in selected}
        total = sum(inv.values())
        weights = {m: inv[m] / total for m in selected}
    else:
        weights = {m: 1.0 / len(selected) for m in selected}

    # ------------------------------------------------------------------
    # 4. Refit selected models on full history
    # ------------------------------------------------------------------
    refitted: dict[str, Any] = {}
    for model_name in selected:
        model = MODEL_CLASSES[model_name](config)
        if model_name == "xgboost" and xgb_best_params:
            model.params.update(xgb_best_params)
        model.fit(series)
        refitted[model_name] = model

    # ------------------------------------------------------------------
    # 5. Serialize to registry
    # ------------------------------------------------------------------
    state_dir = registry.state_dir(version, state)

    for model_name, model_obj in refitted.items():
        model_obj.save(state_dir / ARTIFACT_NAMES[model_name])

    # ensemble_weights.json
    (state_dir / "ensemble_weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )

    # cv_metrics.json — per-model, per-fold + fold details for backtest
    cv_data = {}
    for model_name in MODEL_CLASSES:
        cv_data[model_name] = {
            "folds": fold_scores[model_name],
            "mean_rmse": mean_rmse[model_name],
        }
    # Fold-level predictions/actuals/dates for the /backtest endpoint
    cv_data["fold_details"] = []
    for fi in range(n_folds):
        detail: dict[str, Any] = {
            "dates": fold_dates[fi],
            "actuals": [float(v) for v in fold_actuals[fi]],
            "predictions": {},
        }
        for model_name in MODEL_CLASSES:
            detail["predictions"][model_name] = [float(v) for v in fold_preds[model_name][fi]]
        cv_data["fold_details"].append(detail)

    (state_dir / "cv_metrics.json").write_text(
        json.dumps(cv_data, indent=2, default=str), encoding="utf-8"
    )

    # per-state metadata.json
    registry.write_metadata(version, state, {
        "selected_models": selected,
        "weights": weights,
        "mean_rmse": {m: mean_rmse[m] for m in selected},
        "all_model_mean_rmse": mean_rmse,
    })

    logger.info(
        "[%s] Done — top-%d: %s, weights: %s",
        state,
        top_k,
        selected,
        {m: f"{w:.3f}" for m, w in weights.items()},
    )

    return {
        "state": state,
        "selected": selected,
        "weights": weights,
        "mean_rmse": mean_rmse,
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """Top-level orchestrator: parallel walk-forward CV for all requested states."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.registry = Registry(config["artifacts"]["registry_path"])

    def run(
        self,
        state_series: dict[str, pd.Series],
        states: list[str] | None = None,
        n_jobs: int = -1,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Run the full pipeline for requested states.

        Args:
            state_series: ``{state_name: weekly pd.Series}`` from Preprocessor.
            states: List of states to train, or ``None`` for all.
            n_jobs: Number of parallel workers (``-1`` = all cores).

        Returns:
            ``(version, results_list)`` — version string and per-state summary dicts.
        """
        from joblib import Parallel, delayed

        if states is None:
            states = sorted(state_series.keys())
        else:
            missing = set(states) - set(state_series.keys())
            if missing:
                raise ValueError(f"Unknown states: {missing}")

        version = self.registry.new_version()
        logger.info("Training pipeline started — version=%s, %d states, n_jobs=%d", version, len(states), n_jobs)

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_train_one_state)(
                state, state_series[state], self.config, self.registry, version
            )
            for state in states
        )

        # --- Top-level metadata.json ---
        meta = {
            "version": version,
            "n_states": len(results),
            "states": {},
        }
        for r in results:
            meta["states"][r["state"]] = {
                "selected": r["selected"],
                "weights": r["weights"],
                "mean_rmse": {m: round(v) for m, v in r["mean_rmse"].items()},
            }

        meta_path = self.registry.root / version / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

        # --- manifest ---
        self.registry.write_manifest(version, {"n_states": len(results)})

        logger.info("Pipeline complete — version=%s", version)
        return version, results
