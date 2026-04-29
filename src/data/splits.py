"""Phase 1 — Walk-forward CV generator.

Produces expanding-window train / fixed-window test folds for time-series
cross-validation.  **No leakage:** every test fold starts strictly after
the last train date.

Usage::

    from src.data.splits import walk_forward_splits
    for fold_idx, (train, test) in enumerate(walk_forward_splits(series, cfg)):
        # train: pd.Series (expanding)
        # test:  pd.Series (fixed 8-week window)
"""

from __future__ import annotations

from typing import Any, Iterator

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def walk_forward_splits(
    series: pd.Series,
    config: dict[str, Any] | None = None,
    *,
    initial_train_weeks: int | None = None,
    horizon: int | None = None,
    step: int | None = None,
    max_folds: int | None = None,
    min_folds: int | None = None,
) -> Iterator[tuple[pd.Series, pd.Series]]:
    """Yield ``(train, test)`` pairs from an expanding-window walk-forward split.

    Args:
        series: Weekly time series (must already be resampled to ``W-SUN``).
        config: Project config dict; CV params are read from ``config['cv']``.
            Individual keyword args override values in *config* when both given.
        initial_train_weeks: Minimum number of weeks in the first training window.
        horizon: Number of weeks in each test window (fixed).
        step: Number of weeks the train window advances per fold.
        max_folds: Maximum number of folds to produce.
        min_folds: Raise if fewer than this many folds are possible.

    Yields:
        ``(train_series, test_series)`` — both are ``pd.Series`` slices of
        the original series, preserving the ``DatetimeIndex``.

    Raises:
        ValueError: if the series is too short to produce ``min_folds`` folds.
    """
    # --- resolve parameters --------------------------------------------------
    cv = (config or {}).get("cv", {})
    initial_train_weeks = initial_train_weeks or cv.get("initial_train_weeks", 104)
    horizon = horizon or cv.get("horizon", 8)
    step = step or cv.get("step", 4)
    max_folds = max_folds or cv.get("max_folds", 5)
    min_folds = min_folds or cv.get("min_folds", 3)

    n = len(series)

    # --- generate folds ------------------------------------------------------
    fold_count = 0
    train_end = initial_train_weeks          # first split point (exclusive index)

    while train_end + horizon <= n and fold_count < max_folds:
        train = series.iloc[:train_end]
        test = series.iloc[train_end : train_end + horizon]

        # Sanity: test dates must be strictly after train dates
        assert test.index.min() > train.index.max(), (
            f"Leakage! test starts {test.index.min()} but train ends {train.index.max()}"
        )

        yield train, test
        fold_count += 1
        train_end += step

    if fold_count < min_folds:
        raise ValueError(
            f"Series of length {n} only produced {fold_count} fold(s); "
            f"need at least {min_folds}. "
            f"(initial={initial_train_weeks}, horizon={horizon}, step={step})"
        )

    logger.debug(
        "walk_forward_splits → %d folds (initial=%d, horizon=%d, step=%d, n=%d)",
        fold_count,
        initial_train_weeks,
        horizon,
        step,
        n,
    )
