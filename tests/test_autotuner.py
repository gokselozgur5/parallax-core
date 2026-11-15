from pathlib import Path
import pytest
from parallax.core.autotuner import (
    SCHEMA_VERSION,
    find_crossover,
    get_default_cache_dir,
    get_default_cache_path,
    load_cache,
    measure_latency,
    save_cache,
    validate_cache_schema,
)


def test_default_cache_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("PARALLAX_CACHE_DIR", str(tmp_path))
    assert get_default_cache_dir() == tmp_path
    assert get_default_cache_path() == tmp_path / "autotune.json"
    assert SCHEMA_VERSION == 1


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


def test_cache_save_and_load_roundtrip(tmp_path):
    target = tmp_path / "custom" / "cache.json"
    assert load_cache(target) is None
    expected = {"gemm": 4096, "softmax": 1024, "rmsnorm": 2048}
    saved_path = save_cache(expected, path=target)
    assert saved_path == target
    assert saved_path.is_file()
    loaded = load_cache(target)
    assert loaded == expected


def test_load_cache_corrupt_schema(tmp_path):
    target = tmp_path / "bad.json"
    target.write_text('{"version": 1, "thresholds": "invalid"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or invalid 'thresholds'"):
        load_cache(target)


class DummyFastBackend:
    def gemm(self, a, b):
        return a @ b

    def softmax(self, x, axis=-1):
        return x

    def rmsnorm(self, x, weight, eps=1e-5):
        return x * weight


class DummyRefBackend:
    def gemm(self, a, b):
        return a @ b

    def softmax(self, x, axis=-1):
        return x

    def rmsnorm(self, x, weight, eps=1e-5):
        return x * weight


def test_find_crossover_measurement():
    fast = DummyFastBackend()
    ref = DummyRefBackend()
    lat = measure_latency(fast, "gemm", 16, repeat=2, warmup=1)
    assert lat >= 0
    crossover = find_crossover("gemm", fast, ref, ladder=(8, 16))
    assert crossover in (64, 256)
