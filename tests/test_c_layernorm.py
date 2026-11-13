from __future__ import annotations

import numpy as np
import pytest
from parallax.native.layernorm import c_layernorm, is_layernorm_available


def test_layernorm_availability() -> None:
    assert is_layernorm_available()


def test_layernorm_invalid_dtype() -> None:
    with pytest.raises(TypeError):
        c_layernorm(np.ones((2, 4), dtype=np.float64))


def test_layernorm_zero_rows() -> None:
    x = np.zeros((0, 8), dtype=np.float32)
    out = c_layernorm(x)
    assert out.shape == (0, 8)


def test_layernorm_parity_nn() -> None:
    from parallax.nn import LayerNorm
    from parallax.tensor import Tensor

    rng = np.random.default_rng(42)
    dim = 64
    ln = LayerNorm(dim, eps=1e-5)
    x = rng.standard_normal((16, dim)).astype(np.float32)

    expected = ln(Tensor(x)).data
    actual = c_layernorm(x, weight=ln.weight.data, bias=ln.bias.data, eps=1e-5)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
