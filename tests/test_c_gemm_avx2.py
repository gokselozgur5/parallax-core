from __future__ import annotations

import numpy as np
import pytest

from parallax.backends.cpuinfo import detect_cpu
from parallax.native.gemm_avx2 import (
    gemm_avx2,
    is_avx2_gemm_available,
)


@pytest.fixture(autouse=True)
def require_avx2() -> None:
    cpu = detect_cpu()
    if not cpu.avx2:
        pytest.skip("AVX2 is not supported on this host")
    if not is_avx2_gemm_available():
        pytest.skip("c_gemm_avx2_f32 symbol is not available")
