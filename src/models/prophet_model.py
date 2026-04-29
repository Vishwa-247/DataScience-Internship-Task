"""Phase 4 — Facebook Prophet wrapper.

Uniform interface: ``fit(series)`` / ``predict(horizon)`` / ``save(path)`` / ``load(path)``.

Prophet expects a DataFrame with ``ds`` (date) and ``y`` (value) columns.
This wrapper handles the rename transparently.  US holidays are added
via Prophet's built-in ``add_country_holidays('US')``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from prophet import Prophet

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ProphetModel:
    """Prophet wrapper conforming to the project's uniform model interface."""

    name: str = "prophet"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["models"]["prophet"]
        self.yearly_seasonality: bool = cfg.get("yearly_seasonality", True)
        self.weekly_seasonality: bool = cfg.get("weekly_seasonality", True)
        self.daily_seasonality: bool = cfg.get("daily_seasonality", False)
        self.holidays_country: str = cfg.get("holidays_country", "US")
        self._freq: str = config["data"].get("freq", "W-SUN")
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None

    # -- uniform interface ----------------------------------------------------

    def fit(self, series: pd.Series) -> ProphetModel:
        """Fit Prophet on a weekly ``pd.Series`` with ``DatetimeIndex``.

        Internally converts to ``ds`` / ``y`` format.
        """
        logger.info(
            "Prophet fit: %d obs (%s → %s)",
            len(series),
            series.index.min().date(),
            series.index.max().date(),
        )

        train_df = pd.DataFrame({"ds": series.index, "y": series.values})

        # Suppress Prophet's own verbose stdout
        prophet_logger = logging.getLogger("prophet")
        cmdstan_logger = logging.getLogger("cmdstanpy")
        old_prophet = prophet_logger.level
        old_cmdstan = cmdstan_logger.level
        prophet_logger.setLevel(logging.WARNING)
        cmdstan_logger.setLevel(logging.WARNING)

        try:
            self._model = Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
            )
            self._model.add_country_holidays(country_name=self.holidays_country)
            self._model.fit(train_df)
        finally:
            prophet_logger.setLevel(old_prophet)
            cmdstan_logger.setLevel(old_cmdstan)

        self._last_date = series.index[-1]
        logger.info("Prophet fitted successfully")
        return self

    def predict(self, horizon: int = 8) -> pd.Series:
        """Forecast *horizon* steps ahead.

        Returns:
            ``pd.Series`` with a ``DatetimeIndex`` of future ``W-SUN`` dates,
            containing the ``yhat`` point forecast.
        """
        if self._model is None:
            raise RuntimeError("Must call fit() before predict()")

        future_idx = pd.date_range(
            start=self._last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        future_df = pd.DataFrame({"ds": future_idx})
        forecast = self._model.predict(future_df)

        # Slice to exactly the horizon rows we requested
        yhat = forecast.set_index("ds")["yhat"].loc[future_idx]
        return pd.Series(yhat.values, index=future_idx, name="yhat")

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted model with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.debug("Saved ProphetModel to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProphetModel:
        """Deserialize a previously saved model."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ProphetModel, got {type(obj).__name__}")
        logger.debug("Loaded ProphetModel from %s", path)
        return obj
