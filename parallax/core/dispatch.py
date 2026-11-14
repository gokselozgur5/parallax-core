"""Heuristic platform dispatch policy for selecting optimal backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PlatformInfo:
    """Hardware platform and CPU/accelerator capabilities."""

    arch: str
    system: str
    neon: bool = False
    avx2: bool = False
    cuda: bool = False


DEFAULT_THRESHOLDS: dict[str, int] = {
    "gemm": 64 * 64,
    "matmul": 64 * 64,
    "softmax": 4096,
    "rmsnorm": 4096,
    "attention": 32 * 64,
}
