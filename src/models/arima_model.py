"""Phase 3 — ARIMA / SARIMA wrapper.

Uniform interface: ``fit(series)`` / ``predict(horizon)`` / ``save(path)`` / ``load(path)``.

Uses ``pmdarima.auto_arima`` with ``seasonal=True, m=52, stepwise=True``
for automatic ``(p,d,q)(P,D,Q,52)`` order selection on weekly data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pmdarima as pm

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ARIMAModel:
    """SARIMA wrapper with auto order selection via pmdarima."""

    name: str = "arima"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["models"]["arima"]
        self.seasonal: bool = cfg.get("seasonal", True)
        self.m: int = cfg.get("m", 52)
        self.stepwise: bool = True
        self._model: pm.ARIMA | None = None
        self._freq: str = config["data"].get("freq", "W-SUN")

    # -- uniform interface ----------------------------------------------------

    def fit(self, series: pd.Series) -> ARIMAModel:
        """Fit auto_arima on a weekly ``pd.Series`` with ``DatetimeIndex``.

        Args:
            series: Target values indexed by ``W-SUN`` dates.  Must already be
                    preprocessed (no NaN, resampled).
        """
        n = len(series)
        logger.info(
            "ARIMA fit: %d obs (%s → %s), seasonal=%s, m=%d",
            n,
            series.index.min().date(),
            series.index.max().date(),
            self.seasonal,
            self.m,
        )
        self._model = self._auto_fit(series.values, n)
        self._last_index = series.index[-1]
        return self

    def predict(self, horizon: int = 8) -> pd.Series:
        """Forecast *horizon* steps ahead.

        Returns:
            ``pd.Series`` with a ``DatetimeIndex`` of future ``W-SUN`` dates.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        fc = self._model.predict(n_periods=horizon)
        future_idx = pd.date_range(
            start=self._last_index + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        return pd.Series(fc, index=future_idx, name="yhat")

    # -- internal fitting with fallback chain -----------------------------------

    def _auto_fit(self, values, n: int) -> pm.ARIMA:
        """Try SARIMA(D=0) → non-seasonal ARIMA, logging each attempt.

        m=52 with ≤ 2*m obs causes the OCSB seasonal differencing test to fail.
        Fix D=0 to skip that test while still letting auto_arima choose P and Q.
        """
        if self.seasonal and n > self.m:
            try:
                model = pm.auto_arima(
                    values,
                    seasonal=True,
                    m=self.m,
                    D=0,                       # skip seasonal differencing (too few cycles)
                    stepwise=self.stepwise,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
                logger.info(
                    "ARIMA order: %s, seasonal order: %s (D fixed to 0)",
                    model.order,
                    model.seasonal_order,
                )
                return model
            except Exception as exc:
                logger.warning("SARIMA(D=0) failed (%s), falling back to non-seasonal", exc)

        # Fallback: plain ARIMA (no seasonality)
        model = pm.auto_arima(
            values,
            seasonal=False,
            stepwise=self.stepwise,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        logger.info("ARIMA order: %s (non-seasonal fallback)", model.order)
        return model

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted model with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.debug("Saved ARIMAModel to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ARIMAModel:
        """Deserialize a previously saved model."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ARIMAModel, got {type(obj).__name__}")
        logger.debug("Loaded ARIMAModel from %s", path)
        return obj
