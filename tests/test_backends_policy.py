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


@pytest.mark.parametrize("cpu,expected", CASES)
@pytest.mark.parametrize("op", sorted(MIN_ELEMENTS))
def test_order_above_threshold(op, cpu, expected):
    assert backend_order(op, MIN_ELEMENTS[op], cpu) == expected


@pytest.mark.parametrize("cpu,_expected", CASES)
@pytest.mark.parametrize("op", sorted(MIN_ELEMENTS))
def test_small_inputs_stay_on_numpy(op, cpu, _expected):
    assert backend_order(op, MIN_ELEMENTS[op] - 1, cpu) == ("numpy",)


def test_unknown_op_raises():
    with pytest.raises(ValueError, match="unknown op"):
        backend_order("conv3d", 1 << 20, CASES[0][0])
