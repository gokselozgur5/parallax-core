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
