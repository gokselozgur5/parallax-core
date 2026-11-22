"""Markdown backend availability report."""

from __future__ import annotations

from typing import Any
import numpy as np

from parallax.backends import cpuinfo
from parallax.backends import registry
from parallax.backends.base import Backend
from parallax.backends.parity import check_parity


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
    if isinstance(backend, bool):
        return backend
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


def _evaluate_parity(name: str, backend: Any, avail: bool) -> str:
    if not avail:
        return "n/a"
    if isinstance(backend, Backend):
        try:
            a = np.ones((2, 2), dtype=np.float32)
            b = np.ones((2, 2), dtype=np.float32)
            report = check_parity("gemm", a, b, backends=[backend])
            res = report.get(backend.name) or report.get(name)
            if res is not None:
                if res.status == "ok":
                    return "pass"
                if res.status in ("flagged", "error"):
                    return "fail"
        except Exception:
            pass
    return "not checked"


def generate_markdown_report(
    *,
    cpu_info: cpuinfo.CpuInfo | None = None,
    backends_override: dict[str, Any] | None = None,
    parity_override: dict[str, str] | None = None,
) -> str:
    """Generate a compact markdown report of backends and capabilities."""
    info = cpu_info or cpuinfo.detect_cpu()
    isa = _format_isa_flags(info)
    lines: list[str] = [
        "| Backend | Available | ISA Flags | Parity |",
        "|:---|:---|:---|:---|",
    ]
    items = _collect_backend_items(backends_override)
    for name, b in items:
        av = _check_backend_available(b)
        if parity_override is not None:
            par = parity_override.get(name, "pass" if av else "n/a")
        else:
            par = _evaluate_parity(name, b, av)
        lines.append(_format_row(name, av, isa, par))
    return "\n".join(lines)
