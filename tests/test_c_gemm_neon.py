"""Tests for NEON 8x8 SGEMM microkernel."""

import platform
import numpy as np
import pytest

if platform.machine() not in ("arm64", "aarch64"):
    pytest.skip("NEON GEMM requires arm64/aarch64 architecture", allow_module_level=True)

from parallax.native.gemm_neon import gemm_neon, is_gemm_neon_available


def test_gemm_neon_availability():
    assert is_gemm_neon_available()


@pytest.mark.parametrize("m,k,n", [(8, 8, 8), (16, 16, 16), (3, 5, 7), (9, 11, 13), (17, 23, 31)])
def test_gemm_neon_numerical_parity(m: int, k: int, n: int):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    expected = a @ b
    actual = gemm_neon(a, b)
    assert np.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_gemm_neon_dimension_validation():
    with pytest.raises(ValueError):
        gemm_neon(np.ones((2, 3), dtype=np.float32), np.ones((4, 5), dtype=np.float32))
