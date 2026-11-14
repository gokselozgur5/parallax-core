"""Tests for NEON FlashAttention-1 single-head microkernel."""

import platform
import numpy as np
import pytest

if platform.machine() not in ("arm64", "aarch64"):
    pytest.skip("NEON FlashAttention requires arm64/aarch64", allow_module_level=True)

from parallax.native.flashattn_neon import (
    is_flashattn_neon_available,
    flashattn_neon,
)


def test_flashattn_neon_symbol_available():
    assert is_flashattn_neon_available()


def test_flashattn_neon_invalid_dims():
    q = np.ones((4, 16), dtype=np.float32)
    k = np.ones((4, 32), dtype=np.float32)
    v = np.ones((4, 32), dtype=np.float32)
    with pytest.raises(ValueError):
        flashattn_neon(q, k, v)
