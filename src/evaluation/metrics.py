"""Phase 7 — Pointwise metrics for fold-level scoring.

All functions: ``(y_true, y_pred) -> float``.
MAPE and SMAPE handle zeros in the denominator by skipping or capping those weeks.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%).

    Weeks where ``y_true == 0`` are **excluded** to avoid division by zero.
    Returns ``NaN`` if all actuals are zero.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    Uses the formulation ``200 * |y-ŷ| / (|y| + |ŷ|)``.
    Pairs where both actual and predicted are zero are **excluded**.
    Individual terms are capped at 200 %.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.abs(yt) + np.abs(yp)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    terms = 200.0 * np.abs(yt[mask] - yp[mask]) / denom[mask]
    terms = np.clip(terms, 0, 200)
    return float(np.mean(terms))


def score_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all four metrics in one call."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
