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


def _compute_elements(size_or_shape: int | Sequence[int]) -> int:
    """Compute total element count from an int or shape tuple."""
    if isinstance(size_or_shape, int):
        return max(0, size_or_shape)
    total = 1
    for dim in size_or_shape:
        total *= max(0, dim)
    return total


class DispatchPolicy:
    """Heuristic policy selecting optimal execution backend."""

    def __init__(self, thresholds: dict[str, int] | None = None) -> None:
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds is not None:
            self.thresholds.update(thresholds)
