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
    """Return default cache directory for autotuner data."""
    if "PARALLAX_CACHE_DIR" in os.environ:
        return Path(os.environ["PARALLAX_CACHE_DIR"])
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "parallax"
    return Path.home() / ".cache" / "parallax"


def get_default_cache_path() -> Path:
    """Return default file path for autotune JSON cache."""
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
    "matmul": (16, 32, 64, 128, 256),
    "softmax": (64, 256, 1024, 4096),
    "rmsnorm": (64, 256, 1024, 4096),
}


def _run_op(backend: Any, op: str, size: int) -> None:
    fn = getattr(backend, op, None)
    if fn is None and op == "matmul":
        fn = getattr(backend, "gemm", None)
    if fn is None and op == "gemm":
        fn = getattr(backend, "matmul", None)
    if fn is None:
        return
    if op in ("gemm", "matmul"):
        fn(np.ones((size, size), dtype=np.float32), np.ones((size, size), dtype=np.float32))
    elif op == "softmax":
        fn(np.ones((1, size), dtype=np.float32))
    elif op == "rmsnorm":
        fn(np.ones((1, size), dtype=np.float32), np.ones((size,), dtype=np.float32))


def measure_latency(backend: Any, op: str, size: int, repeat: int = 3, warmup: int = 1) -> float:
    """Measure median duration in nanoseconds for op execution."""
    for _ in range(warmup):
        _run_op(backend, op, size)
    timings = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        _run_op(backend, op, size)
        timings.append(time.perf_counter_ns() - t0)
    return float(np.median(timings))


def find_crossover(op: str, fast: Any, ref: Any, ladder: Sequence[int] | None = None, repeat: int = 3) -> int:
    """Find element count crossover threshold where fast <= ref."""
    steps = ladder or DEFAULT_LADDERS.get(op, (32, 64, 128, 256))
    for s in steps:
        t_fast = measure_latency(fast, op, s, repeat=repeat)
        t_ref = measure_latency(ref, op, s, repeat=repeat)
        n_elems = s * s if op in ("gemm", "matmul") else s
        if t_fast <= t_ref:
            return n_elems
    last = steps[-1]
    return last * last if op in ("gemm", "matmul") else last


def autotune(
    fast: Any, ref: Any, ops: Sequence[str] = ("gemm", "softmax", "rmsnorm"), path: Path | str | None = None
) -> dict[str, int]:
    """Tune crossover thresholds and persist to JSON cache."""
    thresholds = {op: find_crossover(op, fast, ref) for op in ops}
    save_cache(thresholds, path=path)
    return thresholds


class Autotuner:
    """Empirical crossover-size autotuner with JSON cache."""

    def __init__(self, cache_path: Path | str | None = None, repeat: int = 3, warmup: int = 1) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else get_default_cache_path()
        self.repeat = repeat
        self.warmup = warmup

    def get_cache_path(self) -> Path:
        return self.cache_path

    def load_cache(self) -> dict[str, int] | None:
        return load_cache(self.cache_path)

    def save_cache(self, thresholds: dict[str, int]) -> Path:
        return save_cache(thresholds, path=self.cache_path)

    def measure_latency(self, backend: Any, op: str, size: int) -> float:
        return measure_latency(backend, op, size, repeat=self.repeat, warmup=self.warmup)

    def find_crossover(self, op: str, fast: Any, ref: Any, ladder: Sequence[int] | None = None) -> int:
        return find_crossover(op, fast, ref, ladder=ladder, repeat=self.repeat)

    def tune(
        self, fast: Any, ref: Any, ops: Sequence[str] = ("gemm", "softmax", "rmsnorm"), save: bool = True
    ) -> dict[str, int]:
        thresholds = {op: self.find_crossover(op, fast, ref) for op in ops}
        if save:
            self.save_cache(thresholds)
        return thresholds

    def apply_to_policy(self, policy: Any = None) -> Any:
        thresholds = self.load_cache() or {}
        if policy is None:
            from parallax.core.dispatch import DispatchPolicy

            return DispatchPolicy(thresholds=thresholds)
        policy.thresholds.update(thresholds)
        return policy
