import pytest
from parallax.backends.autotune import (
    SCHEMA_VERSION,
    get_default_cache_dir,
    get_default_cache_path,
    validate_cache_schema,
)


def test_default_cache_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLAX_CACHE_DIR", str(tmp_path))
    assert get_default_cache_dir() == tmp_path
    assert get_default_cache_path() == tmp_path / "autotune.json"


def test_validate_cache_schema_valid():
    valid = {"version": SCHEMA_VERSION, "thresholds": {"gemm": 1024}}
    assert validate_cache_schema(valid) is True


def test_validate_cache_schema_invalid():
    with pytest.raises(ValueError, match="Cache root must be a dict"):
        validate_cache_schema("invalid")
    with pytest.raises(ValueError, match="Unsupported schema version"):
        validate_cache_schema({"version": 999, "thresholds": {}})
    with pytest.raises(ValueError, match="Missing or invalid 'thresholds'"):
        validate_cache_schema({"version": SCHEMA_VERSION, "thresholds": None})
    with pytest.raises(ValueError, match="Invalid threshold entry"):
        validate_cache_schema({"version": SCHEMA_VERSION, "thresholds": {"gemm": -1}})
