"""Command-line utility to display Parallax backend availability and capabilities."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from parallax.backends import cpuinfo, registry
from parallax.backends.base import Backend
from parallax.backends.c_backend import CBackend
from parallax.backends.metal_backend import MetalBackend
from parallax.backends.numpy_backend import NumpyBackend
from parallax.backends.report import generate_markdown_report
from parallax.backends.rust_backend import RustBackend


def register_standard_backends() -> None:
    """Ensure standard backends are registered if missing."""
    factories: dict[str, Callable[[], Backend]] = {
        "numpy": NumpyBackend,
        "c": CBackend,
        "rust": RustBackend,
        "metal": MetalBackend,
    }
    existing = set(registry.list_backends())
    for name, factory in factories.items():
        if name not in existing:
            try:
                registry.register(name, factory())
            except Exception:
                pass


def _check_available(backend: Any) -> bool:
    if isinstance(backend, bool):
        return backend
    if isinstance(backend, Backend):
        return bool(backend.available())
    if hasattr(backend, "available") and callable(backend.available):
        return bool(backend.available())
    return True

