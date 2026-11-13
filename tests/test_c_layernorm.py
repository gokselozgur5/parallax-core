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
