"""Markdown backend availability report."""

from __future__ import annotations

from typing import Any
from parallax.backends import cpuinfo
from parallax.backends import registry
from parallax.backends.base import Backend


def _format_isa_flags(info: cpuinfo.CpuInfo) -> str:
    flags: list[str] = []
    if info.neon:
        flags.append("NEON")
    if info.avx2:
        flags.append("AVX2")
    if info.avx512f:
        flags.append("AVX-512")
    return ", ".join(flags) if flags else "none"


def _check_backend_available(backend: Any) -> bool:
    if isinstance(backend, Backend):
        return bool(backend.available())
    if hasattr(backend, "available") and callable(backend.available):
        return bool(backend.available())
    return True
