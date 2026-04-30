"""Phase 5 — XGBoost direct multi-step (h=1…8) with Optuna HPO + SHAP.

Uniform interface: ``fit(series)`` / ``predict(horizon)`` / ``save(path)`` / ``load(path)``.

**Direct multi-step:** trains 8 separate ``XGBRegressor`` models, one per
forecast horizon step.  For step *h*, the target is ``y[t+h]`` paired with
features ``X[t]``.

**Optuna HPO:** ``optimize(train_series, val_series)`` runs 50 trials on a
single fold (typically fold 1).  Best params are stored and reused by
subsequent ``fit()`` calls on all folds.

**SHAP:** ``explain()`` returns SHAP values from a ``shap.Explainer`` (PermutationExplainer)
on the *h = 1* model only. Uses ``shap.maskers.Independent`` for XGBoost 3.x compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.engineer import FeatureEngineer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostModel:
    """Direct multi-step XGBoost with Optuna HPO and SHAP explanations."""

    name: str = "xgboost"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["models"]["xgboost"]
        self.multi_step: str = cfg.get("multi_step", "direct")
        self.n_trials: int = cfg.get("optuna_trials", 50)
        self.early_stopping: int = cfg.get("early_stopping_rounds", 50)
        self._config = config
        self._freq: str = config["data"].get("freq", "W-SUN")
        self._horizon: int = config["forecast"]["horizon_weeks"]
        self._seed: int = config.get("seed", 42)

        # Default hyper-parameters (overridden by optimize())
        self.params: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "random_state": self._seed,
        }

        self._models: dict[int, xgb.XGBRegressor] = {}
        self._engineer: FeatureEngineer | None = None
        self._last_X: pd.DataFrame | None = None
        self._train_X: pd.DataFrame | None = None   # kept for SHAP
        self._last_index: pd.Timestamp | None = None

    # -- uniform interface ----------------------------------------------------

    def fit(self, series: pd.Series) -> XGBoostModel:
        """Train 8 direct-step XGBoost models on features from *series*.

        The ``FeatureEngineer`` is fitted internally; its state is stored for
        test-time ``predict()``.
        """
        self._engineer = FeatureEngineer(self._config)
        X, y = self._engineer.fit_transform(series)
        self._train_X = X
        self._last_X = X.iloc[[-1]]
        self._last_index = series.index[-1]

        for h in range(1, self._horizon + 1):
            X_h = X.iloc[: len(X) - h]
            y_h = y.iloc[h:]
            model = xgb.XGBRegressor(**self.params)
            model.fit(X_h, y_h)
            self._models[h] = model

        logger.info(
            "XGBoost fit: %d direct models (h=1…%d), %d features, %d train rows",
            self._horizon,
            self._horizon,
            X.shape[1],
            len(X),
        )
        return self

    def predict(self, horizon: int = 8) -> pd.Series:
        """Predict *horizon* steps from the last known feature row.

        Returns:
            ``pd.Series`` with a ``W-SUN`` ``DatetimeIndex``.
        """
        if not self._models:
            raise RuntimeError("Must call fit() before predict()")

        preds = [
            self._models[h].predict(self._last_X)[0]
            for h in range(1, horizon + 1)
        ]
        future_idx = pd.date_range(
            start=self._last_index + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        return pd.Series(preds, index=future_idx, name="yhat")

    # -- Optuna HPO -----------------------------------------------------------

    def optimize(
        self,
        train_series: pd.Series,
        val_series: pd.Series,
        n_trials: int | None = None,
    ) -> XGBoostModel:
        """Run Optuna HPO on a single fold (typically fold 1).

        Trains all 8 direct models per trial, evaluates 8-step RMSE against
        *val_series*, and stores the best params for future ``fit()`` calls.
        """
        import optuna

        n_trials = n_trials or self.n_trials
        eng = FeatureEngineer(self._config)
        X_train, y_train = eng.fit_transform(train_series)

        horizon = min(len(val_series), self._horizon)
        val_actual = val_series.values[:horizon]

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self._seed,
            }
            preds: list[float] = []
            for h in range(1, horizon + 1):
                X_h = X_train.iloc[: len(X_train) - h]
                y_h = y_train.iloc[h:]
                mdl = xgb.XGBRegressor(**params)
                mdl.fit(X_h, y_h)
                preds.append(mdl.predict(X_train.iloc[[-1]])[0])

            rmse = float(np.sqrt(np.mean((np.array(preds) - val_actual) ** 2)))
            return rmse

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self._seed))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        self.params.update(study.best_params)
        self.params["random_state"] = self._seed

        logger.info(
            "Optuna HPO complete: %d trials, best 8-step RMSE=%.0f\n  params=%s",
            n_trials,
            study.best_value,
            study.best_params,
        )
        return self

    # -- SHAP -----------------------------------------------------------------

    def explain(self, X: pd.DataFrame | None = None) -> np.ndarray:
        """Compute SHAP values for the h=1 model via ``shap.Explainer`` (PermutationExplainer).

        Args:
            X: Feature matrix to explain.  Defaults to full training features.

        Returns:
            SHAP values array with shape ``(n_samples, n_features)``.
        """
        import shap

        if 1 not in self._models:
            raise RuntimeError("Must fit() before explain()")

        if X is None:
            X = self._train_X

        # Use predict callable to avoid shap/xgboost 3.x base_score parsing bug
        explainer = shap.Explainer(self._models[1].predict, shap.maskers.Independent(X))
        explanation = explainer(X)
        shap_values = explanation.values
        logger.info(
            "SHAP values computed (h=1 model), shape: %s",
            shap_values.shape if hasattr(shap_values, "shape") else "N/A",
        )
        return shap_values

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.debug("Saved XGBoostModel to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> XGBoostModel:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected XGBoostModel, got {type(obj).__name__}")
        logger.debug("Loaded XGBoostModel from %s", path)
        return obj
