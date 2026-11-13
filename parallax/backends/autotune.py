"""Crossover-size autotuner with JSON cache."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def get_default_cache_dir() -> Path:
    """Return the default cache directory for autotune data."""
    if "PARALLAX_CACHE_DIR" in os.environ:
        return Path(os.environ["PARALLAX_CACHE_DIR"])
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "parallax"
    return Path.home() / ".cache" / "parallax"


def get_default_cache_path() -> Path:
    """Return default path to the autotune JSON cache file."""
    return get_default_cache_dir() / "autotune.json"


def validate_cache_schema(data: Any) -> bool:
    """Validate autotune cache JSON structure."""
    if not isinstance(data, dict):
        raise ValueError("Cache root must be a dict")
    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {data.get('version')}")
    thresholds = data.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Missing or invalid 'thresholds' dict")
    for op, val in thresholds.items():
        if not isinstance(op, str) or not isinstance(val, int) or val < 0:
            raise ValueError(f"Invalid threshold entry: {op!r}: {val!r}")
    return True


def save_cache(thresholds: dict[str, int], path: Path | str | None = None) -> Path:
    """Save crossover thresholds to JSON cache."""
    target = Path(path) if path is not None else get_default_cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "thresholds": dict(thresholds),
    }
    validate_cache_schema(payload)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target


def load_cache(path: Path | str | None = None) -> dict[str, int] | None:
    """Load and validate crossover thresholds from JSON cache."""
    target = Path(path) if path is not None else get_default_cache_path()
    if not target.is_file():
        return None
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    validate_cache_schema(data)
    return data["thresholds"]


DEFAULT_LADDERS: dict[str, tuple[int, ...]] = {
    "gemm": (16, 32, 64, 128, 256),
    "softmax": (64, 256, 1024, 4096),
    "rmsnorm": (64, 256, 1024, 4096),
}


def _run_op(backend: Any, op: str, size: int) -> None:
    if op == "gemm":
        a = np.ones((size, size), dtype=np.float32)
        b = np.ones((size, size), dtype=np.float32)
        backend.gemm(a, b)
    elif op == "softmax":
        x = np.ones((1, size), dtype=np.float32)
        backend.softmax(x)
    elif op == "rmsnorm":
        x = np.ones((1, size), dtype=np.float32)
        w = np.ones((size,), dtype=np.float32)
        backend.rmsnorm(x, w)


