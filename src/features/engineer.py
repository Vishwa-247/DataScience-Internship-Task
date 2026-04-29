"""Phase 2 — FeatureEngineer with strict fit-on-train-only semantics.

**No-leakage contract:**

1. ``fit(train_series)``  — stores the *tail* of the training series
   (length = ``max(lags)`` which is 30 for the default config) so that
   test-time lag / rolling features can be computed without seeing
   future data.
2. ``fit_transform(train_series)``  — fits, builds features, then **drops**
   the warmup rows (first ~30) that are NaN due to lag/rolling creation.
   Returns ``(X_train, y_train)`` with aligned indices.
3. ``transform(test_series)``  — prepends the stored tail, builds features
   for test rows only, and **flags an error** if any test row has NaN
   (that would indicate a bug, since the tail should cover all lags).

Serializable with ``joblib.dump``/``joblib.load``.
"""

from __future__ import annotations

from typing import Any

import holidays as holidays_lib
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Build lag / rolling / time / holiday / Fourier features from a weekly series.

    All state is captured in ``fit()`` so the object can be serialised alongside
    the model for reproducible inference.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["features"]
        self.lags: list[int] = cfg["lags"]                          # [1, 7, 30]
        self.rolling_windows: list[int] = cfg["rolling_windows"]    # [4, 8, 13]
        self.rolling_stats: list[str] = cfg["rolling_stats"]        # [mean, std]
        self.fourier_k: int = cfg["fourier_terms"]                  # 3
        self.use_holidays: bool = cfg["holidays"]                   # True
        self.time_features: list[str] = cfg["time_features"]        # [week_of_year, month, quarter, year]

        # Computed in fit()
        self._fitted: bool = False
        self._train_tail: pd.Series | None = None
        self._warmup: int = 0                                       # rows lost to lag/rolling

    # -- public API -----------------------------------------------------------

    def fit(self, train_series: pd.Series) -> FeatureEngineer:
        """Store the tail of *train_series* needed for test-time feature building.

        Nothing is learned from the data (no statistics that leak into test).
        The tail length equals ``max(max_lag, max_rolling_window - 1)``.
        """
        max_lag = max(self.lags) if self.lags else 0
        max_roll = (max(self.rolling_windows) - 1) if self.rolling_windows else 0
        self._warmup = max(max_lag, max_roll)

        self._train_tail = train_series.iloc[-self._warmup :].copy()
        self._fitted = True

        logger.debug(
            "fit: warmup=%d (max_lag=%d, max_roll_gap=%d), tail stored %d rows",
            self._warmup,
            max_lag,
            max_roll,
            len(self._train_tail),
        )
        return self

    def fit_transform(self, train_series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Fit on *train_series* and return ``(X_train, y_train)`` with warmup rows dropped.

        Returns:
            X_train: Feature DataFrame, index aligned to remaining rows.
            y_train: Target Series matching X_train's index.
        """
        self.fit(train_series)
        X = self._build_features(train_series, prepend_tail=False)

        n_before = len(X)
        X = X.dropna()
        n_dropped = n_before - len(X)

        if n_dropped > 0:
            logger.debug(
                "fit_transform: dropped %d/%d train warmup rows (expected ~%d)",
                n_dropped,
                n_before,
                self._warmup,
            )

        y = train_series.loc[X.index]
        return X, y

    def transform(self, series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Build features for *series* (typically test), using stored train tail.

        Returns:
            X: Feature DataFrame for the requested rows.
            y: Target Series matching X's index.

        Logs an **ERROR** if any output row contains NaN — that signals a bug
        (the stored tail should supply enough history for all lags/rolling).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() or fit_transform() before transform()")

        X = self._build_features(series, prepend_tail=True)

        nan_rows = X.isna().any(axis=1).sum()
        if nan_rows > 0:
            nan_cols = X.columns[X.isna().any()].tolist()
            logger.error(
                "BUG: %d test rows have NaN in columns %s. "
                "This should not happen — check tail length (%d) vs warmup (%d).",
                nan_rows,
                nan_cols,
                len(self._train_tail),
                self._warmup,
            )

        y = series.loc[X.index]
        return X, y

    # -- internals ------------------------------------------------------------

    def _build_features(self, series: pd.Series, *, prepend_tail: bool) -> pd.DataFrame:
        """Core feature-building logic. Returns a DataFrame indexed only on *series*."""
        if prepend_tail and self._train_tail is not None:
            work = pd.concat([self._train_tail, series])
        else:
            work = series.copy()

        original_index = series.index
        df = pd.DataFrame(index=work.index)

        # --- lag features ----------------------------------------------------
        for lag in self.lags:
            df[f"lag_{lag}"] = work.shift(lag)

        # --- rolling statistics ----------------------------------------------
        for window in self.rolling_windows:
            rolled = work.rolling(window)
            for stat in self.rolling_stats:
                if stat == "mean":
                    df[f"rolling_mean_{window}"] = rolled.mean()
                elif stat == "std":
                    df[f"rolling_std_{window}"] = rolled.std()

        # --- time features (deterministic from index) ------------------------
        idx = work.index
        for feat in self.time_features:
            if feat == "week_of_year":
                df["week_of_year"] = idx.isocalendar().week.astype(int).values
            elif feat == "month":
                df["month"] = idx.month
            elif feat == "quarter":
                df["quarter"] = idx.quarter
            elif feat == "year":
                df["year"] = idx.year

        # --- holiday flag (1 if any US federal holiday falls in the week) ----
        if self.use_holidays:
            df["holiday"] = self._holiday_flags(idx)

        # --- Fourier terms for annual seasonality ----------------------------
        if self.fourier_k > 0:
            week_num = idx.isocalendar().week.astype(int).values
            for k in range(1, self.fourier_k + 1):
                df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * week_num / 52.1775)
                df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * week_num / 52.1775)

        # Slice back to only the rows the caller asked about
        df = df.loc[original_index]
        return df

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _holiday_flags(index: pd.DatetimeIndex) -> list[int]:
        """Return 1 for each week-ending date whose Mon–Sun span contains a US holiday."""
        years = range(index.min().year, index.max().year + 1)
        us_hol = holidays_lib.US(years=years)
        flags: list[int] = []
        for week_end in index:
            week_start = week_end - pd.Timedelta(days=6)
            has_holiday = any(
                (week_start + pd.Timedelta(days=d)) in us_hol for d in range(7)
            )
            flags.append(int(has_holiday))
        return flags
