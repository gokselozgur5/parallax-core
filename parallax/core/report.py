"""Markdown backend availability report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class BackendReportRow:
    """Row metadata representing backend availability and capabilities."""

    name: str
    available: bool
    isa_flags: str
    parity: str
    capabilities: tuple[str, ...] = ()


def format_isa_flags(info: Any = None) -> str:
    """Format detected or provided CPU ISA flags into a comma-separated list."""
    if info is None:
        try:
            from parallax.backends import cpuinfo
            info = cpuinfo.detect_cpu()
        except Exception:
            info = None
    flags: list[str] = []
    if getattr(info, "neon", False):
        flags.append("NEON")
    if getattr(info, "avx2", False):
        flags.append("AVX2")
    if getattr(info, "avx512f", False):
        flags.append("AVX-512")
    if getattr(info, "cuda", False):
        flags.append("CUDA")
    return ", ".join(flags) if flags else "none"
