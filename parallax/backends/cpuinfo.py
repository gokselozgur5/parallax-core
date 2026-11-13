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


def normalize_arch(machine: str) -> str:
    """Map platform machine names onto arm64 or x86_64."""
    key = machine.strip().lower()
    return _ARCH_ALIASES.get(key, key)


def parse_sysctl(text: str, machine: str) -> CpuInfo:
    """Build CpuInfo from Darwin ``sysctl`` key/value output."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            values[key.strip()] = value.strip()
    leaves = ("machdep.cpu.features", "machdep.cpu.leaf7_features")
    flags = set(" ".join(values.get(k, "") for k in leaves).upper().split())
    return CpuInfo(
        arch=normalize_arch(machine),
        system="Darwin",
        neon="1" in (values.get("hw.optional.neon"), values.get("hw.optional.AdvSIMD")),
        avx2="AVX2" in flags or values.get("hw.optional.avx2_0") == "1",
        avx512f="AVX512F" in flags or values.get("hw.optional.avx512f") == "1",
    )
