"""AVX-512 FlashAttention-1 single-head microkernel binding."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np
from parallax.native.loader import bind

_FN_FLASHATTN_AVX512: Optional[Callable[..., Any]] = None
_FN_HAS_AVX512_FLASHATTN: Optional[Callable[..., Any]] = None
