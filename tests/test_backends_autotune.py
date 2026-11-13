import pytest
from parallax.backends.autotune import (
    SCHEMA_VERSION,
    autotune,
    find_crossover,
    get_default_cache_dir,
    get_default_cache_path,
    load_cache,
    save_cache,
    validate_cache_schema,
)
from parallax.backends.numpy_backend import NumpyBackend


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
    name = "dummy_fast"

    def gemm(self, a, b):
        return a @ b

    def softmax(self, x, axis=-1):
        return x

    def rmsnorm(self, x, weight, eps=1e-5):
        return x * weight


def test_find_crossover_and_autotune(tmp_path):
    fast = DummyFastBackend()
    ref = NumpyBackend()
    threshold = find_crossover("gemm", fast, ref, ladder=(8, 16))
    assert threshold > 0
    cache_file = tmp_path / "autotune.json"
    res = autotune(fast, ref, ops=("gemm", "softmax"), path=cache_file)
    assert "gemm" in res and "softmax" in res
    assert cache_file.is_file()


