from __future__ import annotations

import sys
from typing import Any
from parallax.backends.base import Backend


class MetalBackend(Backend):
    name: str = "metal"

    def available(self) -> bool:
        return sys.platform == "darwin"

    def supports(self, op: str, dtype: Any, shape: tuple[int, ...]) -> bool:
        return False
