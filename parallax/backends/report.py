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


def _format_row(name: str, avail: bool, isa: str, parity: str) -> str:
    av_str = "true" if avail else "false"
    return f"| {name} | {av_str} | {isa} | {parity} |"


def _collect_backend_items(override: dict[str, Any] | None) -> list[tuple[str, Any]]:
    if override is not None:
        return sorted(override.items(), key=lambda x: x[0])
    names = registry.list_backends()
    return sorted([(n, registry.get(n)) for n in names], key=lambda x: x[0])


def generate_markdown_report(
    *,
    cpu_info: cpuinfo.CpuInfo | None = None,
    backends_override: dict[str, Any] | None = None,
    parity_override: dict[str, str] | None = None,
) -> str:
    """Generate a compact markdown report of backends and capabilities."""
    info = cpu_info or cpuinfo.probe()
    isa = _format_isa_flags(info)
    lines: list[str] = [
        "| Backend | Available | ISA Flags | Parity |",
        "|:---|:---|:---|:---|",
    ]
    items = _collect_backend_items(backends_override)
    for name, b in items:
        av = _check_backend_available(b)
        par = (
            parity_override.get(name, "pass" if av else "n/a")
            if parity_override is not None
            else ("pass" if av else "n/a")
        )
        lines.append(_format_row(name, av, isa, par))
    return "\n".join(lines)
