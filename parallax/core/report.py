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


def _check_backend_available(backend: Any) -> bool:
    if isinstance(backend, bool):
        return backend
    if hasattr(backend, "available") and callable(backend.available):
        try:
            return bool(backend.available())
        except Exception:
            return False
    return True


def _collect_backend_items(override: dict[str, Any] | None) -> list[tuple[str, Any]]:
    if override is not None:
        return sorted(override.items(), key=lambda x: str(x[0]))
    try:
        from parallax.backends import registry
        names = registry.list_backends()
        return sorted([(n, registry.get(n)) for n in names], key=lambda x: str(x[0]))
    except Exception:
        return []


COMMON_OPS: tuple[str, ...] = ("gemm", "matmul", "softmax", "rmsnorm", "attention")


def _detect_capabilities(backend: Any, override: Sequence[str] | None = None) -> tuple[str, ...]:
    if override is not None:
        return tuple(override)
    if hasattr(backend, "capabilities"):
        caps = backend.capabilities
        return tuple(caps() if callable(caps) else caps)
    if hasattr(backend, "supports") and callable(backend.supports):
        detected: list[str] = []
        for op in COMMON_OPS:
            try:
                if backend.supports(op):
                    detected.append(op)
            except Exception:
                pass
        return tuple(detected)
    return ()


def collect_backend_rows(
    backends: dict[str, Any] | None = None,
    *,
    cpu_info: Any = None,
    parity: dict[str, str] | None = None,
    capabilities: dict[str, Sequence[str]] | None = None,
) -> list[BackendReportRow]:
    """Collect status, ISA flags, parity and capabilities for backends."""
    isa = format_isa_flags(cpu_info)
    items = _collect_backend_items(backends)
    rows: list[BackendReportRow] = []
    for name, b in items:
        av = _check_backend_available(b)
        par = parity.get(name, "pass" if av else "n/a") if parity else ("pass" if av else "n/a")
        cap_override = capabilities.get(name) if capabilities else None
        caps = _detect_capabilities(b, cap_override)
        rows.append(BackendReportRow(name=name, available=av, isa_flags=isa, parity=par, capabilities=caps))
    return rows


def format_markdown_table(rows: Sequence[BackendReportRow], *, include_capabilities: bool = False) -> str:
    """Format backend report rows into a clean Markdown table."""
    headers = ["Backend", "Available", "ISA Flags", "Parity"]
    if include_capabilities:
        headers.append("Capabilities")
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join([":---"] * len(headers)) + " |"
    lines = [header_line, sep_line]
    for r in rows:
        av_str = "true" if r.available else "false"
        cols = [r.name, av_str, r.isa_flags, r.parity]
        if include_capabilities:
            caps_str = ", ".join(r.capabilities) if r.capabilities else "none"
            cols.append(caps_str)
        lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(lines)


def generate_markdown_report(
    *,
    cpu_info: Any = None,
    backends: dict[str, Any] | None = None,
    backends_override: dict[str, Any] | None = None,
    parity_override: dict[str, str] | None = None,
    capabilities: dict[str, Sequence[str]] | None = None,
    include_capabilities: bool = False,
) -> str:
    """Generate a clean markdown report of active backends and capabilities."""
    target_backends = backends if backends is not None else backends_override
    inc_caps = include_capabilities or (capabilities is not None)
    rows = collect_backend_rows(
        target_backends,
        cpu_info=cpu_info,
        parity=parity_override,
        capabilities=capabilities,
    )
    return format_markdown_table(rows, include_capabilities=inc_caps)



__all__ = [
    "BackendReportRow",
    "collect_backend_rows",
    "format_isa_flags",
    "format_markdown_table",
    "generate_markdown_report",
]
