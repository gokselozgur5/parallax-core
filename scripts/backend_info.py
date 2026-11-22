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
from parallax.backends.report import _evaluate_parity, generate_markdown_report
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


def collect_backend_info_dict() -> dict[str, Any]:
    """Collect system and backend info into a structured dictionary."""
    register_standard_backends()
    info = cpuinfo.detect_cpu()
    backends_data: dict[str, dict[str, Any]] = {}
    for name in sorted(registry.list_backends()):
        b = registry.get(name)
        avail = _check_available(b)
        backends_data[name] = {
            "available": avail,
            "parity": _evaluate_parity(name, b, avail),
        }
    return {
        "cpu": {
            "arch": info.arch,
            "system": info.system,
            "neon": info.neon,
            "avx2": info.avx2,
            "avx512f": info.avx512f,
        },
        "backends": backends_data,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for backend_info."""
    parser = argparse.ArgumentParser(
        description="Display Parallax backend availability and capabilities."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit output as valid JSON instead of markdown table.",
    )
    args = parser.parse_args(argv)

    register_standard_backends()
    if args.json_output:
        data = collect_backend_info_dict()
        print(json.dumps(data, indent=2))
    else:
        report = generate_markdown_report()
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
