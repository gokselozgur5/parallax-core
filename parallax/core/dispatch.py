"""Heuristic platform dispatch policy for selecting optimal backend.

Canonical dispatch authority resides in `parallax.backends`.
"""

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

    def candidate_backends(
        self,
        op: str,
        size_or_shape: int | Sequence[int] = 0,
        platform: PlatformInfo | None = None,
    ) -> list[str]:
        """Return preferred backend candidate chain for op and size."""
        n_elems = _compute_elements(size_or_shape)
        min_elems = self.thresholds.get(op, 0)
        if n_elems < min_elems:
            return ["numpy"]
        if platform is None:
            return ["numpy"]
        if platform.cuda:
            return ["cuda", "numpy"]
        arch = platform.arch.lower()
        if arch in ("arm64", "aarch64") and platform.system == "Darwin":
            return ["metal", "c-neon", "numpy"]
        if arch in ("arm64", "aarch64") and platform.neon:
            return ["c-neon", "numpy"]
        if arch in ("x86_64", "amd64") and platform.avx2:
            return ["c-avx", "rust", "numpy"]
        if arch in ("x86_64", "amd64"):
            return ["rust", "numpy"]
        return ["numpy"]

    def select_backend(
        self,
        op: str,
        size_or_shape: int | Sequence[int] = 0,
        platform: PlatformInfo | None = None,
        available_backends: Sequence[str] | set[str] | None = None,
    ) -> str:
        """Select the optimal available backend according to heuristic policy.

        Raises:
            ValueError: If ``available_backends`` contains none of the candidates.
        """
        candidates = self.candidate_backends(op, size_or_shape, platform=platform)
        if available_backends is None:
            return candidates[0]
        avail_set = set(available_backends)
        for backend in candidates:
            if backend in avail_set:
                return backend
        msg = f"No candidate backend for {op!r} in available_backends: {candidates}"
        raise ValueError(msg)


_DEFAULT_POLICY = DispatchPolicy()


def select_optimal_backend(
    op: str,
    size_or_shape: int | Sequence[int] = 0,
    platform: PlatformInfo | None = None,
    available_backends: Sequence[str] | set[str] | None = None,
) -> str:
    """Convenience helper to select optimal backend with default policy."""
    return _DEFAULT_POLICY.select_backend(
        op,
        size_or_shape,
        platform=platform,
        available_backends=available_backends,
    )


def get_candidate_backends(
    op: str,
    size_or_shape: int | Sequence[int] = 0,
    platform: PlatformInfo | None = None,
) -> list[str]:
    """Convenience helper to get candidate backends with default policy."""
    return _DEFAULT_POLICY.candidate_backends(op, size_or_shape, platform=platform)
