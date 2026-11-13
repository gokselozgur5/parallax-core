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
