"""Smoke tests — fast, no model training, verifies core contracts."""
import numpy as np
import pandas as pd
import pytest


# -- helpers ------------------------------------------------------------------

def _make_weekly_series(n: int = 120, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    values = rng.uniform(1e7, 1e9, size=n)
    return pd.Series(values, index=idx, name="Total")


# -- Test 1: config loads -----------------------------------------------------

def test_config_loads():
    from src.data.loader import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    for key in ("data", "preprocessing", "models", "cv", "forecast", "artifacts"):
        assert key in cfg, f"Missing top-level key: {key}"
    assert cfg["forecast"]["horizon_weeks"] == 8
    assert cfg["cv"]["min_folds"] >= 1


# -- Test 2: preprocessor output shape + no NaN -------------------------------

def test_preprocessor_no_nan():
    from src.data.loader import load_config, load_sales
    from src.data.preprocessor import Preprocessor
    cfg = load_config()
    df = load_sales(config=cfg)
    result = Preprocessor(cfg).transform(df)
    assert len(result) == 43, f"Expected 43 states, got {len(result)}"
    for state, series in result.items():
        assert series.isna().sum() == 0, f"{state} has NaN values after preprocessing"
        assert len(series) == 256, f"{state} has {len(series)} weeks, expected 256"
        assert series.index.freq is not None, f"{state} index has no freq"


# -- Test 3: walk-forward no leakage ------------------------------------------

def test_walk_forward_no_leakage():
    from src.data.loader import load_config
    from src.data.splits import walk_forward_splits
    cfg = load_config()
    series = _make_weekly_series(n=256)
    folds = list(walk_forward_splits(series, cfg))
    assert len(folds) >= cfg["cv"]["min_folds"], "Too few folds produced"
    for i, (train, test) in enumerate(folds):
        assert test.index.min() > train.index.max(), (
            f"Fold {i}: data leakage! test starts {test.index.min()} "
            f"but train ends {train.index.max()}"
        )
        assert len(test) == cfg["cv"]["horizon"], (
            f"Fold {i}: test length {len(test)} != horizon {cfg['cv']['horizon']}"
        )


# -- Test 4: score_forecast returns correct keys ------------------------------

def test_score_forecast_keys():
    from src.evaluation.metrics import score_forecast
    rng = np.random.default_rng(0)
    actual = rng.uniform(1e7, 1e9, size=8)
    predicted = actual * rng.uniform(0.9, 1.1, size=8)
    result = score_forecast(actual, predicted)
    for key in ("rmse", "mae", "mape", "smape"):
        assert key in result, f"Missing metric: {key}"
        assert result[key] >= 0, f"{key} is negative: {result[key]}"
    assert result["rmse"] >= result["mae"], "RMSE should be >= MAE"


# -- Test 5: ARIMAModel has uniform interface ---------------------------------

def test_arima_interface():
    from src.models.arima_model import ARIMAModel
    from src.data.loader import load_config
    cfg = load_config()
    model = ARIMAModel(cfg)
    for method in ("fit", "predict", "save", "load"):
        assert hasattr(model, method), f"ARIMAModel missing method: {method}"
    assert model.name == "arima"


# -- Test 6: LSTM raises on short series -------------------------------------

def test_lstm_short_series_guard():
    from src.models.lstm_model import LSTMModel
    from src.data.loader import load_config
    cfg = load_config()
    model = LSTMModel(cfg)
    short = _make_weekly_series(n=10)
    with pytest.raises(ValueError, match="LSTMModel needs at least"):
        model.fit(short)


# -- Test 7: walk_forward_splits skips gracefully on very short series --------

def test_splits_short_series_raises():
    from src.data.loader import load_config
    from src.data.splits import walk_forward_splits
    cfg = load_config()
    short = _make_weekly_series(n=50)
    with pytest.raises(ValueError):
        list(walk_forward_splits(short, cfg))
