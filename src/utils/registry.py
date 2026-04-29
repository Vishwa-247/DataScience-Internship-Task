"""Versioned model registry.

Layout::

    artifacts/registry/
      manifest.json                  # {current: vTS, versions: [{version, ...}]}
      v20260101_103045/
        states/
          California/
            arima.joblib
            prophet.joblib
            xgboost.joblib
            lstm.pt
            feature_engineer.joblib
            metadata.json            # {selected_models, weights, metrics}
          Texas/
            ...

Designed to satisfy Phase 7 (training pipeline writes) and Phase 8
(API lazy-loads from the latest version).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


class Registry:
    """Versioned save/load for per-state model bundles."""

    MANIFEST = "manifest.json"
    METADATA = "metadata.json"

    def __init__(self, root: str | Path = "artifacts/registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- versioning -----------------------------------------------------------

    @staticmethod
    def _stamp() -> str:
        return "v" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def new_version(self) -> str:
        """Create and return a new ``vYYYYMMDD_HHMMSS`` version directory."""
        version = self._stamp()
        (self.root / version / "states").mkdir(parents=True, exist_ok=True)
        return version

    # -- per-state paths ------------------------------------------------------

    def state_dir(self, version: str, state: str) -> Path:
        d = self.root / version / "states" / state
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- save -----------------------------------------------------------------

    def save_artifact(self, version: str, state: str, name: str, obj: Any) -> Path:
        """Pickle ``obj`` to ``<state_dir>/<name>``. Returns the path written."""
        path = self.state_dir(version, state) / name
        joblib.dump(obj, path)
        return path

    def write_metadata(self, version: str, state: str, metadata: dict) -> Path:
        """Write per-state metadata JSON (selected models, weights, metrics)."""
        path = self.state_dir(version, state) / self.METADATA
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return path

    def write_manifest(self, version: str, payload: dict | None = None) -> Path:
        """Update top-level manifest pointing ``current`` at ``version``."""
        path = self.root / self.MANIFEST
        existing: dict = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing["current"] = version
        existing.setdefault("versions", []).append(
            {"version": version, "created_at": datetime.now().isoformat(), **(payload or {})}
        )
        path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        return path

    # -- load -----------------------------------------------------------------

    def latest_version(self) -> str | None:
        """Return the version pointed to by the manifest, or the newest dir as a fallback."""
        manifest_path = self.root / self.MANIFEST
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            current = data.get("current")
            if current and (self.root / current).is_dir():
                return current
        versions = sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and p.name.startswith("v")
        )
        return versions[-1] if versions else None

    def load_artifact(self, version: str, state: str, name: str) -> Any:
        return joblib.load(self.state_dir(version, state) / name)

    def read_metadata(self, version: str, state: str) -> dict:
        path = self.state_dir(version, state) / self.METADATA
        return json.loads(path.read_text(encoding="utf-8"))

    def list_states(self, version: str) -> list[str]:
        states_dir = self.root / version / "states"
        if not states_dir.exists():
            return []
        return sorted(p.name for p in states_dir.iterdir() if p.is_dir())
