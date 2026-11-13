import pytest

from parallax.backends.scope import (
    clear_scopes,
    current_backend,
    scoped_backend,
    use_backend,
)


@pytest.fixture(autouse=True)
def _reset_scopes():
    clear_scopes()
    yield
    clear_scopes()


def test_default_scope_is_none():
    assert current_backend() is None
    assert scoped_backend("gemm") is None


def test_simple_scope_and_cleanup():
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


def test_per_op_scope_and_global_fallback():
    with use_backend("numpy"):
        assert current_backend("gemm") == "numpy"
        with use_backend("metal", op="gemm"):
            assert current_backend("gemm") == "metal"
            assert current_backend("softmax") == "numpy"
        assert current_backend("gemm") == "numpy"


def test_clear_override_with_none():
    with use_backend("c"):
        assert current_backend() == "c"
        with use_backend(None):
            assert current_backend() is None
        assert current_backend() == "c"


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

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f1 = executor.submit(worker, "t1", "c-neon")
        f2 = executor.submit(worker, "t2", "rust")
        f3 = executor.submit(worker, "t3", "metal")
        for f in (f1, f2, f3):
            f.result()

    assert results["t1"] == (None, "c-neon", None)
    assert results["t2"] == (None, "rust", None)
    assert results["t3"] == (None, "metal", None)
    assert current_backend() is None
