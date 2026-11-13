from __future__ import annotations

import sys
from typing import Any
import numpy as np
from parallax.backends.base import Backend
from parallax.metal_engine import is_metal_available, metal_gemm, metal_rmsnorm


class MetalBackend(Backend):
    name: str = "metal"

    def available(self) -> bool:
        if sys.platform != "darwin":
            return False
        return is_metal_available()
