from __future__ import annotations

import numpy as np
import pytest
from parallax.native.rope import is_rope_available, rope_in_place


def test_rope_availability() -> None:
    assert is_rope_available()


def test_rope_invalid_shape_and_dtype() -> None:
    with pytest.raises(ValueError):
        rope_in_place(np.zeros((4, 8), dtype=np.float32))

    with pytest.raises(ValueError):
        rope_in_place(np.zeros((2, 4, 7), dtype=np.float32))

    with pytest.raises(TypeError):
        rope_in_place(np.zeros((2, 4, 8), dtype=np.float64))


def test_rope_zero_length() -> None:
    x = np.zeros((0, 4, 8), dtype=np.float32)
    out = rope_in_place(x)
    assert out.shape == (0, 4, 8)
