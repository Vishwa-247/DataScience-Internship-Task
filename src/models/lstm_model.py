"""Phase 6 — LSTM (PyTorch) seq-to-vec, lookback=52, multi-step direct output (8).

Uniform interface: ``fit(series)`` / ``predict(horizon)`` / ``save(path)`` / ``load(path)``.

Architecture::

    LSTM(input_size=n_features, hidden=64, num_layers=2, batch_first=True)
    → Linear(64, 8)

Predicts all 8 horizon steps in one forward pass.

**MinMaxScaler** is fitted on the *train target only* (not features) before
building per-timestep features.  The LSTM lookback window (52 weeks) replaces
explicit lag / rolling features — instead each time step carries:
``[scaled_value, week_of_year, month, quarter, year, holiday,
fourier_sin×3, fourier_cos×3]`` = 12 features.

Early stopping with patience=5 on a 10 % validation split held out from the
end of the training series.  ``save()`` persists ``state_dict + scaler``
together via joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import holidays as holidays_lib
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PyTorch network
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    """LSTM → Linear producing *horizon* outputs in one forward pass."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, input_size)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]          # (batch, hidden)
        return self.fc(last_hidden)           # (batch, horizon)


# ---------------------------------------------------------------------------
# Wrapper with uniform interface
# ---------------------------------------------------------------------------

class LSTMModel:
    """LSTM wrapper conforming to the project's uniform model interface."""

    name: str = "lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["models"]["lstm"]
        self.lookback: int = cfg["lookback"]          # 52
        self.hidden: int = cfg["hidden"]              # 64
        self.num_layers: int = cfg["layers"]          # 2
        self.dropout: float = cfg["dropout"]          # 0.1
        self.epochs: int = cfg["epochs"]              # 50
        self.patience: int = cfg["patience"]          # 5
        self.batch_size: int = cfg["batch_size"]      # 16
        self.lr: float = cfg["lr"]                    # 0.001
        self._horizon: int = config["forecast"]["horizon_weeks"]  # 8
        self._freq: str = config["data"].get("freq", "W-SUN")
        self._seed: int = config.get("seed", 42)

        self._scaler: MinMaxScaler | None = None
        self._net: _LSTMNet | None = None
        self._device: torch.device | None = None
        self._n_features: int | None = None
        self._last_window: np.ndarray | None = None   # (lookback, n_features)
        self._last_index: pd.Timestamp | None = None

    # -- uniform interface ----------------------------------------------------

    def fit(self, series: pd.Series) -> LSTMModel:
        """Scale target → build per-timestep features → window → train LSTM.

        A 10 % validation split is held out from the **end** of the training
        windows for early stopping.
        """
        from src.utils.seed import set_seed
        set_seed(self._seed)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. MinMaxScaler on target only, fit on train
        self._scaler = MinMaxScaler()
        scaled = self._scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        scaled_series = pd.Series(scaled, index=series.index)

        # 2. Per-timestep features (no lags — lookback handles temporal context)
        features = self._build_timestep_features(scaled_series)
        self._n_features = features.shape[1]

        # 3. Sliding windows
        X_all, y_all = self._create_windows(features, scaled)
        n_total = len(X_all)
        n_val = max(1, int(n_total * 0.1))
        X_train, y_train = X_all[:-n_val], y_all[:-n_val]
        X_val, y_val = X_all[-n_val:], y_all[-n_val:]

        logger.info(
            "LSTM fit: %d obs → %d windows (train=%d, val=%d), %d features, "
            "lookback=%d, device=%s",
            len(series), n_total, len(X_train), len(X_val),
            self._n_features, self.lookback, self._device,
        )

        # 4. Build network
        self._net = _LSTMNet(
            input_size=self._n_features,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self._horizon,
        ).to(self._device)

        # 5. Train with early stopping
        self._train_loop(X_train, y_train, X_val, y_val)

        # 6. Store last window for predict()
        self._last_window = features[-self.lookback:]
        self._last_index = series.index[-1]
        return self

    def predict(self, horizon: int = 8) -> pd.Series:
        """Forward pass on stored last window → inverse-scale → pd.Series."""
        if self._net is None:
            raise RuntimeError("Must call fit() before predict()")

        self._net.eval()
        X = torch.tensor(self._last_window, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred_scaled = self._net(X).cpu().numpy().flatten()[:horizon]

        pred = self._scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        future_idx = pd.date_range(
            start=self._last_index + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        return pd.Series(pred, index=future_idx, name="yhat")

    # -- training loop --------------------------------------------------------

    def _train_loop(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self._device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self._device)
        X_v = torch.tensor(X_val, dtype=torch.float32).to(self._device)
        y_v = torch.tensor(y_val, dtype=torch.float32).to(self._device)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_ctr = 0
        best_state: dict | None = None

        for epoch in range(self.epochs):
            # --- train ---
            self._net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X_train)

            # --- validate ---
            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_v), y_v).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr = 0
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    logger.info("Early stop at epoch %d (best val MSE=%.6f)", epoch + 1, best_val_loss)
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
            self._net.to(self._device)

        final_epoch = epoch + 1 if 'epoch' in dir() else self.epochs
        logger.info("LSTM training done: %d epochs, best val MSE=%.6f", final_epoch, best_val_loss)

    # -- feature / window helpers ---------------------------------------------

    @staticmethod
    def _build_timestep_features(scaled_series: pd.Series) -> np.ndarray:
        """Build per-timestep feature matrix: scaled value + time + holiday + Fourier.

        Returns array of shape ``(n, 12)`` — no warmup NaNs since there are
        no lag / rolling features.
        """
        idx = scaled_series.index
        vals = scaled_series.values.reshape(-1, 1)

        # Time features normalised to [0, 1]
        week = idx.isocalendar().week.astype(int).values.reshape(-1, 1) / 52.0
        month = (idx.month.values.reshape(-1, 1) - 1) / 11.0
        quarter = (idx.quarter.values.reshape(-1, 1) - 1) / 3.0
        yr = idx.year.values
        yr_norm = ((yr - yr.min()) / max(1, yr.max() - yr.min())).reshape(-1, 1)

        # Holiday flag
        us_hol = holidays_lib.US(years=range(idx.min().year, idx.max().year + 1))
        hol = np.array([
            float(any((dt - pd.Timedelta(days=d)) in us_hol for d in range(7)))
            for dt in idx
        ]).reshape(-1, 1)

        # Fourier (K = 3)
        wk = idx.isocalendar().week.astype(int).values
        fourier = []
        for k in range(1, 4):
            fourier.append(np.sin(2 * np.pi * k * wk / 52.1775).reshape(-1, 1))
            fourier.append(np.cos(2 * np.pi * k * wk / 52.1775).reshape(-1, 1))

        return np.hstack([vals, week, month, quarter, yr_norm, hol] + fourier)

    def _create_windows(
        self, features: np.ndarray, scaled_target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sliding windows: ``(lookback, n_feat)`` input → ``(horizon,)`` target."""
        n = len(features)
        X, y = [], []
        for i in range(n - self.lookback - self._horizon + 1):
            X.append(features[i : i + self.lookback])
            y.append(scaled_target[i + self.lookback : i + self.lookback + self._horizon])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save ``state_dict + scaler`` together with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "state_dict": {k: v.cpu() for k, v in self._net.state_dict().items()} if self._net else None,
            "scaler": self._scaler,
            "n_features": self._n_features,
            "last_window": self._last_window,
            "last_index": self._last_index,
            "lookback": self.lookback,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "horizon": self._horizon,
            "freq": self._freq,
            "seed": self._seed,
        }
        joblib.dump(state, path)
        logger.debug("Saved LSTMModel to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> LSTMModel:
        """Reconstruct model from saved ``state_dict + scaler``."""
        state = joblib.load(path)
        config = {
            "models": {"lstm": {
                "lookback": state["lookback"],
                "hidden": state["hidden"],
                "layers": state["num_layers"],
                "dropout": state["dropout"],
                "epochs": 0, "patience": 0, "batch_size": 1, "lr": 0,
            }},
            "forecast": {"horizon_weeks": state["horizon"]},
            "data": {"freq": state["freq"]},
            "seed": state.get("seed", 42),
        }
        obj = cls(config)
        obj._scaler = state["scaler"]
        obj._n_features = state["n_features"]
        obj._last_window = state["last_window"]
        obj._last_index = state["last_index"]
        obj._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj._net = _LSTMNet(
            input_size=state["n_features"],
            hidden_size=state["hidden"],
            num_layers=state["num_layers"],
            dropout=state["dropout"],
            horizon=state["horizon"],
        ).to(obj._device)
        obj._net.load_state_dict(state["state_dict"])
        obj._net.eval()
        logger.debug("Loaded LSTMModel from %s", path)
        return obj
