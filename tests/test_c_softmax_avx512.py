import numpy as np
import pytest
from parallax.native.softmax_avx512 import (
    _get_softmax_avx512_fn,
    is_avx512_softmax_available,
    softmax_avx512,
)


def test_softmax_avx512_symbols_bound():
    fn = _get_softmax_avx512_fn()
    assert fn is not None
    assert isinstance(is_avx512_softmax_available(), bool)


def test_softmax_avx512_shape_checks():
    x = np.array(1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="Input array must have at least 1 dimension"):
        softmax_avx512(x)


def test_softmax_avx512_execution_or_safe_guard():
    x = np.random.randn(4, 32).astype(np.float32)
    if is_avx512_softmax_available():
        res = softmax_avx512(x)
        expected = np.exp(x - np.max(x, axis=-1, keepdims=True))
        expected /= np.sum(expected, axis=-1, keepdims=True)
        assert np.allclose(res, expected, atol=1e-4)
    else:
        with pytest.raises(RuntimeError, match="AVX-512F is not supported on this CPU"):
            softmax_avx512(x)


def test_softmax_avx512_parity_skipped_when_unavailable():
    if not is_avx512_softmax_available():
        pytest.skip("AVX-512F ISA not supported on this host")
    x = np.random.randn(3, 49).astype(np.float32)
    res = softmax_avx512(x)
    expected = np.exp(x - np.max(x, axis=-1, keepdims=True))
    expected /= np.sum(expected, axis=-1, keepdims=True)
    assert np.allclose(res, expected, atol=1e-4)

