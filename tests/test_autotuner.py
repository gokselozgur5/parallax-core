import time
import pytest
from parallax.core.autotuner import (
    SCHEMA_VERSION,
    Autotuner,
    autotune,
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


def test_autotune_orchestrator(tmp_path):
    fast = DummyFastBackend()
    ref = DummyRefBackend()
    cache_path = tmp_path / "custom_tune.json"
    res = autotune(fast, ref, ops=("gemm", "softmax"), path=cache_path)
    assert "gemm" in res and "softmax" in res
    assert cache_path.is_file()


def test_autotuner_class_workflow(tmp_path):
    tuner = Autotuner(cache_path=tmp_path / "tuner.json", repeat=2)
    assert tuner.load_cache() is None
    fast = DummyFastBackend()
    ref = DummyRefBackend()
    thresholds = tuner.tune(fast, ref, ops=("gemm",))
    assert "gemm" in thresholds
    assert tuner.get_cache_path().is_file()
    assert tuner.load_cache() == thresholds


def test_autotuner_apply_to_policy(tmp_path):
    cache_file = tmp_path / "autotune.json"
    save_cache({"gemm": 128, "softmax": 512}, path=cache_file)
    tuner = Autotuner(cache_path=cache_file)
    policy = tuner.apply_to_policy()
    assert policy.thresholds["gemm"] == 128
    assert policy.thresholds["softmax"] == 512


class SlowBackend:
    def gemm(self, a, b):
        time.sleep(0.001)
        return a @ b


def test_find_crossover_slow_backend_fallback():
    slow = SlowBackend()
    fast = DummyFastBackend()
    crossover = find_crossover("gemm", slow, fast, ladder=(8, 16), repeat=1)
    assert crossover == 256
