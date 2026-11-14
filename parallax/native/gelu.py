"""Python wrapper for the native C GELU kernel."""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Optional
import numpy as np

from parallax.native.loader import bind

_GELU_FN: Optional[Callable[..., Any]] = None
_GELU_INIT_ATTEMPTED: bool = False
