from __future__ import annotations
from typing import Any, Optional
import parallax.backends.patch as _bp
from parallax.backends.linear import DispatchedLinear
import parallax.nn as nn

_ORIG_LINEAR: Optional[type] = None


def patch_linear() -> None:
    global _ORIG_LINEAR
    if _ORIG_LINEAR is None:
        _ORIG_LINEAR = nn.Linear
        setattr(nn, "Linear", DispatchedLinear)  # noqa: B010


def unpatch_linear() -> None:
    global _ORIG_LINEAR
    if _ORIG_LINEAR is not None:
        setattr(nn, "Linear", _ORIG_LINEAR)  # noqa: B010
        _ORIG_LINEAR = None


def is_patched() -> bool:
    return _ORIG_LINEAR is not None


def enable_native(model: Any = None) -> Any:
    return patch_linear() if model is None else _bp.enable_native(model)


def disable_native(model: Any = None) -> Any:
    return unpatch_linear() if model is None else _bp.disable_native(model)
