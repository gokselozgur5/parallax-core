from __future__ import annotations
from typing import Any
import parallax.backends.patch as _bp


def enable_native(model: Any = None) -> Any:
    return _bp.enable_native(model)


def disable_native(model: Any = None) -> Any:
    return _bp.disable_native(model)
