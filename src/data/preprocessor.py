"""Phase 1 — Per-state weekly resampling, imputation, and IQR outlier capping.

**Critical rule:** dates in the raw data are irregular (gaps from 1 to 91 days).
Every state series MUST be resampled to ``W-SUN`` with
``resample('W-SUN').sum(min_count=1)`` *before* any other processing.

``min_count=1`` ensures gap-weeks produce ``NaN`` (not ``0``), since the
original data contains zero genuine zero-value rows.  Subsequent linear
interpolation then fills only the true gaps.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Resample irregular dates to W-SUN, impute gaps, and cap outliers.

    The entire pipeline is per-state, stateless, and deterministic:
        raw DataFrame  →  ``dict[state_name, pd.Series]``  (weekly, gap-free)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.freq: str = config["data"]["freq"]                          # W-SUN
        self.date_col: str = config["data"].get("date_col", "Date")
        self.target_col: str = config["data"].get("target_col", "Total")
        self.group_col: str = config["data"].get("group_col", "State")
        self.imputation: str = config["preprocessing"]["imputation"]     # linear
        self.outlier_strategy: str = config["preprocessing"]["outlier_strategy"]  # cap | none
        self.iqr_k: float = config["preprocessing"]["iqr_multiplier"]    # 1.5

    # -- public API -----------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Run the full preprocess pipeline for every state.

        Returns:
            ``{state_name: pd.Series}`` — weekly ``Total`` series indexed by
            ``DatetimeIndex`` (freq = ``W-SUN``), with no NaNs and outliers capped.
        """
        result: dict[str, pd.Series] = {}
        for state, group in df.groupby(self.group_col):
            series = self._resample(group)
            series = self._impute(series)
            if self.outlier_strategy == "cap":
                series = self._iqr_clip(series, k=self.iqr_k)
            series.name = self.target_col
            result[state] = series
            logger.debug(
                "%s — %d raw → %d weekly (%.0f%% gap weeks imputed)",
                state,
                len(group),
                len(series),
                (len(series) - group[self.date_col].nunique()) / len(series) * 100
                if len(series) > 0
                else 0,
            )
        logger.info(
            "Preprocessed %d states → %s freq, %d weeks each (min=%d, max=%d)",
            len(result),
            self.freq,
            next(iter(result.values())).shape[0] if result else 0,
            min(len(s) for s in result.values()) if result else 0,
            max(len(s) for s in result.values()) if result else 0,
        )
        return result

    # -- internals ------------------------------------------------------------

    def _resample(self, group: pd.DataFrame) -> pd.Series:
        """Resample a single state to ``W-SUN``.

        Uses ``sum(min_count=1)`` so gap-weeks become ``NaN``, not ``0``.
        """
        s = (
            group
            .set_index(self.date_col)[self.target_col]
            .sort_index()
            .resample(self.freq)
            .sum(min_count=1)
        )
        return s

    def _impute(self, series: pd.Series) -> pd.Series:
        """Chain: linear interpolation → forward-fill → back-fill.

        Handles interior gaps (linear) and any leading/trailing NaN (ffill/bfill).
        """
        n_missing = series.isna().sum()
        if n_missing == 0:
            return series
        series = series.interpolate(method=self.imputation)
        series = series.ffill().bfill()
        remaining = series.isna().sum()
        if remaining > 0:
            logger.warning("Still %d NaN after full imputation chain!", remaining)
        return series

    @staticmethod
    def _iqr_clip(series: pd.Series, k: float = 1.5) -> pd.Series:
        """Cap values outside [Q1 - k*IQR, Q3 + k*IQR]."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        clipped = series.clip(lower, upper)
        n_clipped = (series != clipped).sum()
        if n_clipped > 0:
            logger.debug("IQR clip: %d values capped to [%.0f, %.0f]", n_clipped, lower, upper)
        return clipped
