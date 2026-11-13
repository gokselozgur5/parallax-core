"""Crossover-size autotuner with JSON cache."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
