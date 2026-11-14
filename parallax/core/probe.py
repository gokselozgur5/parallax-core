"""Host CPU vector SIMD feature probe."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CpuFeatures:
    """Detected host CPU architecture and vector SIMD capabilities."""

    arch: str
    system: str = ""
    neon: bool = False
    avx2: bool = False
    avx512: bool = False
    avx512f: bool = False

    def __post_init__(self) -> None:
        has_512 = self.avx512 or self.avx512f
        object.__setattr__(self, "avx512", has_512)
        object.__setattr__(self, "avx512f", has_512)

    @property
    def has_neon(self) -> bool:
        return self.neon

    @property
    def has_avx2(self) -> bool:
        return self.avx2

    @property
    def has_avx512(self) -> bool:
        return self.avx512

    def has_feature(self, feature: str) -> bool:
        norm = feature.lower().replace("-", "").replace("_", "")
        return {"neon": self.neon, "avx2": self.avx2, "avx512": self.avx512, "avx512f": self.avx512}.get(norm, False)


_ARCH_MAP = {"aarch64": "arm64", "arm64": "arm64", "amd64": "x86_64", "x86_64": "x86_64"}


def normalize_arch(machine: str) -> str:
    """Map platform machine name to canonical architecture."""
    key = machine.strip().lower()
    return _ARCH_MAP.get(key, key)



CpuProbe = CpuFeatures
CpuInfo = CpuFeatures



def parse_proc_cpuinfo(text: str, machine: str = "") -> CpuFeatures:
    """Build CpuFeatures from Linux /proc/cpuinfo content."""
    flags: set[str] = set()
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in ("flags", "features"):
            flags.update(value.lower().split())
    arch = normalize_arch(machine) if machine else "unknown"
    has_512 = any(f.startswith("avx512") for f in flags)
    return CpuFeatures(
        arch=arch,
        system="Linux",
        neon="asimd" in flags or "neon" in flags,
        avx2="avx2" in flags,
        avx512=has_512,
    )
