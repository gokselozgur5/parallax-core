import numpy as np

from parallax.backends.c_backend import CBackend
from parallax.backends.numpy_backend import NumpyBackend
from parallax.dispatch.gelu_fast import gelu_fast_dispatch


def _c_backend() -> CBackend:
    backend = CBackend()
    assert backend.available(), "C engine is not built on this host"
    return backend


def test_supports_agrees_with_what_the_backend_can_run():
    backend = _c_backend()
    a = np.ones((2, 3, 4), dtype=np.float32)
    b = np.ones((4, 5), dtype=np.float32)
    if backend.supports("gemm", a.dtype, a.shape):
        np.testing.assert_allclose(backend.gemm(a, b), NumpyBackend().gemm(a, b))


def test_rmsnorm_weight_contract_matches_the_numpy_backend():
    x, w = np.ones((2, 8), dtype=np.float32), np.full((1, 8), 2.0, dtype=np.float32)
    np.testing.assert_allclose(
        _c_backend().rmsnorm(x, w), NumpyBackend().rmsnorm(x, w), rtol=1e-5
    )


def test_gelu_fast_dispatch_keeps_the_input_dtype():
    assert gelu_fast_dispatch(np.linspace(-3.0, 3.0, 7)).dtype == np.float64
