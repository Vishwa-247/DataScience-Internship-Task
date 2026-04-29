"""Phase 1 — Excel loader + schema validation.

Handles the mixed-format Date column in sales.xlsx:
  - 40 % of rows carry native ``datetime`` objects (Excel serial dates)
  - 60 % carry ``DD-MM-YYYY`` strings
Both are normalised by ``pd.to_datetime(…, dayfirst=True)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"State", "Date", "Total", "Category"}


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read the project-wide YAML config."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sales(
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load the raw sales Excel file and return a schema-validated DataFrame.

    Args:
        path: Explicit path to the ``.xlsx`` file.  Falls back to
              ``config['data']['path']`` when *None*.
        config: Project config dict.  Loaded from ``config.yaml`` when *None*.

    Returns:
        DataFrame with columns ``[State, Date, Total, Category]`` where
        ``Date`` is ``datetime64[ns]`` and ``Total`` is ``float64``.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or data quality checks fail.
    """
    if config is None:
        config = load_config()

    if path is None:
        path = Path(config["data"]["path"])
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading %s …", path)
    df = pd.read_excel(path)

    # --- schema validation ---------------------------------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # --- parse mixed-format dates --------------------------------------------
    date_col = config["data"].get("date_col", "Date")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)

    # --- type enforcement ----------------------------------------------------
    target_col = config["data"].get("target_col", "Total")
    df[target_col] = df[target_col].astype(float)

    # --- basic data-quality checks -------------------------------------------
    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    if null_counts.any():
        logger.warning("Null values detected:\n%s", null_counts[null_counts > 0])

    n_states = df[config["data"].get("group_col", "State")].nunique()
    n_dates = df[date_col].nunique()
    logger.info(
        "Loaded %d rows — %d states × %d unique dates (%s → %s)",
        len(df),
        n_states,
        n_dates,
        df[date_col].min().date(),
        df[date_col].max().date(),
    )
    return df
