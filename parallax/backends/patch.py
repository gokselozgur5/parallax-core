"""Reversible patcher swapping Linear for DispatchedLinear in models."""

from __future__ import annotations

from typing import Any
from parallax.backends.linear import DispatchedLinear
from parallax.nn import Linear, Module


def _swap(m: Any, src: type, make: Any) -> None:
    for k, v in list(getattr(m, "__dict__", {}).items()):
        if isinstance(v, src) and (src is not Linear or type(v) is Linear):
            setattr(m, k, make(v))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, src) and (src is not Linear or type(x) is Linear):
                    v[i] = make(x)
                elif isinstance(x, Module):
                    _swap(x, src, make)
        elif isinstance(v, Module):
            _swap(v, src, make)


def enable_native(model: Any) -> Any:
    """Recursively swap Linear for DispatchedLinear in-place."""
    if type(model) is Linear:
        return DispatchedLinear.from_linear(model)
    _swap(model, Linear, DispatchedLinear.from_linear)
    return model



def _restore(dl: DispatchedLinear) -> Linear:
    lin = Linear(dl.in_features, dl.out_features, bias=dl.bias is not None)
    lin.weight = dl.weight
    lin.bias = dl.bias
    return lin


def disable_native(model: Any) -> Any:
    """Recursively restore original Linear modules in-place."""
    if isinstance(model, DispatchedLinear):
        return _restore(model)
    _swap(model, DispatchedLinear, _restore)
    return model
