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
