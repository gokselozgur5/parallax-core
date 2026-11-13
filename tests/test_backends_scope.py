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
