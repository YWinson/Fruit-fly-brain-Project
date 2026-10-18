"""Run logging, environment snapshots, and data hashing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .paths import ensure_dir


def _json_safe(value):
    """Recursively replace NaN/Inf with null so the JSON is standards-compliant."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def configure_utf8_stdio() -> None:
    """Make Chinese paths/messages display correctly in UTF-8 terminals."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def make_run_id(prefix: str = "m1") -> str:
    """Create a sortable run id: ``m1_20261018_113000_1234``."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{os.getpid() % 10000:04d}"


def collect_env() -> dict:
    """Collect versions relevant to reproducibility."""
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for name in ("numpy", "pandas", "scipy", "sklearn", "mne", "moabb", "torch"):
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[name] = None
    return env


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file in chunks (for data provenance records)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunLogger:
    """Write a run log plus JSON/YAML snapshots into one run directory."""

    def __init__(self, run_id: str, log_dir: str | Path):
        self.run_id = run_id
        self.log_dir = ensure_dir(Path(log_dir) / run_id)
        self.log_path = self.log_dir / "run.log"
        self._handle = open(self.log_path, "a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(line)
        self._handle.write(line + "\n")
        self._handle.flush()

    def save_json(self, name: str, payload: object) -> Path:
        path = self.log_dir / name
        path.write_text(
            json.dumps(
                _json_safe(payload),
                ensure_ascii=False,
                indent=2,
                default=str,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        return path

    def save_yaml(self, name: str, payload: object) -> Path:
        path = self.log_dir / name
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def save_config(self, config: dict) -> None:
        self.save_json("config.json", config)
        self.save_yaml("config.yaml", config)

    def save_text(self, name: str, text: str) -> Path:
        path = self.log_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()
