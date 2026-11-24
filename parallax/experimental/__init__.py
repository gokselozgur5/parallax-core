"""Experimental and research modules catalog for Parallax.

Isolates paper and research prototypes (alignment, parameter-efficient adapters,
routing, and speculative decoding) from core engine primitives.
"""

from __future__ import annotations

import importlib
from typing import Any

EXPERIMENTAL_MODULES: tuple[str, ...] = (
    # Alignment
    "dpo",
    "ipo",
    "kto",
    "orpo",
    "ppo",
    "simpo",
    # Adapters
    "dora",
    "xlora",
    "qgalore",
    # Routing / speculative
    "medusa",
    "mod",
    "mla",
    "ring_attention",
    "tree_drafting",
    "activation_beacon",
    "ecr",
)


def __getattr__(name: str) -> Any:
    if name in EXPERIMENTAL_MODULES:
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.{name}":
                return importlib.import_module(f"parallax.{name}")
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EXPERIMENTAL_MODULES",
    *EXPERIMENTAL_MODULES,
]
