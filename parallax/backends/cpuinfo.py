"""Host CPU feature probe used by backend dispatch."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

_PROC_CPUINFO = "/proc/cpuinfo"
_ARCH_ALIASES = {"aarch64": "arm64", "arm64": "arm64", "amd64": "x86_64", "x86_64": "x86_64"}


@dataclass(frozen=True)
class CpuInfo:
    """Architecture and SIMD extensions reported by the host."""

    arch: str
    system: str
    neon: bool = False
    avx2: bool = False
    avx512f: bool = False
