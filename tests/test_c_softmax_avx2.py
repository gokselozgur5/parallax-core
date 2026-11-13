from __future__ import annotations

import numpy as np
import pytest

from parallax.backends.cpuinfo import detect_cpu
from parallax.native.avx2_softmax import (
    is_avx2_softmax_available,
    softmax_avx2,
)


@pytest.fixture(autouse=True)
def require_avx2() -> None:
    cpu = detect_cpu()
    if not cpu.avx2:
        pytest.skip("AVX2 is not supported on this host")
    if not is_avx2_softmax_available():
        pytest.skip("c_softmax_avx2_f32 symbol is not available")


def test_avx2_softmax_invalid_dim() -> None:
    x_1d = np.ones((8,), dtype=np.float32)
    with pytest.raises(ValueError, match="input must be a 2D matrix"):
        softmax_avx2(x_1d)


def test_avx2_softmax_invalid_axis() -> None:
    x = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="only supports last dimension"):
        softmax_avx2(x, axis=0)
