from pathlib import Path
import pytest
from parallax.core.autotuner import (
    SCHEMA_VERSION,
    get_default_cache_dir,
    get_default_cache_path,
)


def test_default_cache_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLAX_CACHE_DIR", str(tmp_path))
    assert get_default_cache_dir() == tmp_path
    assert get_default_cache_path() == tmp_path / "autotune.json"
    assert SCHEMA_VERSION == 1
