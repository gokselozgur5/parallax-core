from dataclasses import dataclass

import pytest

from parallax.backends.policy import MIN_ELEMENTS, backend_order


@dataclass(frozen=True)
class FakeCpu:
    arch: str
    system: str
    neon: bool = False
    avx2: bool = False


CASES = [
    (FakeCpu("arm64", "Darwin", neon=True), ("metal", "c-neon", "numpy")),
    (FakeCpu("aarch64", "Linux", neon=True), ("c-neon", "numpy")),
    (FakeCpu("x86_64", "Linux", avx2=True), ("c-avx", "rust", "numpy")),
    (FakeCpu("AMD64", "Windows", avx2=True), ("c-avx", "rust", "numpy")),
    (FakeCpu("x86_64", "Linux"), ("rust", "numpy")),
    (FakeCpu("riscv64", "Linux"), ("numpy",)),
]
