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


CpuProbe = CpuFeatures
CpuInfo = CpuFeatures
