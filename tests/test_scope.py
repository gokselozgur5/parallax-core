"""Tests for thread-local backend scope in parallax.core."""

from __future__ import annotations

import pytest

from parallax.core.scope import (
    clear_scopes,
    current_backend,
    scoped_backend,
    use_backend,
)


def test_use_backend_basic_scope():
    assert current_backend() is None
    with use_backend("numpy") as b:
        assert b == "numpy"
        assert current_backend() == "numpy"
    assert current_backend() is None



def test_nested_scopes_restore_correctly():
    with use_backend("numpy"):
        assert current_backend() == "numpy"
        with use_backend("rust"):
            assert current_backend() == "rust"
        assert current_backend() == "numpy"
    assert current_backend() is None


def test_scope_as_function_decorator():
    @use_backend("c")
    def compute():
        return current_backend()

    assert compute() == "c"
    assert current_backend() is None



def test_per_op_scope_and_none_clearing():
    with use_backend("numpy"):
        with use_backend("metal", op="gemm"):
            assert current_backend("gemm") == "metal"
            assert current_backend("softmax") == "numpy"
        with use_backend(None):
            assert current_backend() is None
        assert current_backend() == "numpy"


def test_scope_restores_on_exception():
    with pytest.raises(RuntimeError, match="failure"):
        with use_backend("rust"):
            assert current_backend() == "rust"
            raise RuntimeError("failure")
    assert current_backend() is None



def test_invalid_backend_and_op_inputs():
    with pytest.raises(ValueError, match="backend cannot be empty"):
        with use_backend("   "):
            pass
    with pytest.raises(TypeError, match="backend must be a str"):
        with use_backend(123):
            pass
    with pytest.raises(ValueError, match="op cannot be empty"):
        with use_backend("c", op="  "):
            pass
    with pytest.raises(TypeError, match="op must be a str"):
        with use_backend("c", op=123):
            pass


import concurrent.futures


def test_thread_isolation_and_no_leakage():
    results = {}

    def worker(name: str, backend_name: str):
        before = current_backend()
        with use_backend(backend_name):
            inside = current_backend()
        after = current_backend()
        results[name] = (before, inside, after)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(worker, "t1", "c-neon")
        f2 = ex.submit(worker, "t2", "rust")
        f1.result()
        f2.result()

    assert results["t1"] == (None, "c-neon", None)
    assert results["t2"] == (None, "rust", None)
    assert current_backend() is None



def test_scoped_backend_alias():
    clear_scopes()
    assert scoped_backend() is None
    with use_backend("rust"):
        assert scoped_backend() == "rust"
    assert scoped_backend() is None


def test_use_backend_scope_selects_the_backend_for_dispatch():
    import numpy as np

    from parallax.backends import registry
    from parallax.backends.base import Backend
    from parallax.backends.dispatch import dispatch
    from parallax.backends.numpy_backend import NumpyBackend
    from parallax.backends.scope import use_backend

    class _Marker(Backend):
        name = "rust"
        available = lambda self: True  # noqa: E731
        supports = lambda self, op, dtype, shape: op == "gemm"  # noqa: E731
        gemm = lambda self, a, b: np.full((1, 1), 42.0)  # noqa: E731

    registry.clear()
    registry.register("numpy", NumpyBackend())
    registry.register("rust", _Marker())
    try:
        a = np.ones((2, 2), dtype=np.float32)
        with use_backend("rust"):
            out = dispatch("gemm", a, a)
    finally:
        registry.clear()
    assert out[0, 0] == 42.0, "dispatch ignored the use_backend scope"


def test_clear_scopes_inside_use_backend_exits_cleanly():
    from parallax.backends.scope import (
        clear_scopes as clear_backend_scopes,
        current_backend as current_scoped_backend,
        use_backend as use_scoped_backend,
    )

    with use_scoped_backend("numpy"):
        assert current_scoped_backend() == "numpy"
        clear_backend_scopes()
        assert current_scoped_backend() is None
    assert current_scoped_backend() is None

    with pytest.raises(RuntimeError, match="caller boom"):
        with use_scoped_backend("numpy"):
            with use_scoped_backend("rust"):
                clear_backend_scopes()
                raise RuntimeError("caller boom")
    assert current_scoped_backend() is None

