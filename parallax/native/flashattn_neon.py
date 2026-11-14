"""Python wrapper for the NEON FlashAttention-1 single-head microkernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional

import numpy as np

from parallax.native.loader import bind

_FN_FLASHATTN_NEON: Optional[Callable[..., Any]] = None
