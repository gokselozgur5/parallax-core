import dataclasses

import pytest

from parallax.backends import cpuinfo
from parallax.backends.cpuinfo import CpuInfo, parse_proc_cpuinfo, parse_sysctl

APPLE_SILICON = "hw.optional.neon: 1\nhw.optional.AdvSIMD: 1\nmachdep.cpu.brand_string: Apple M2\n"
INTEL_MAC = (
    "machdep.cpu.features: FPU SSE SSE2 AVX1.0\n"
    "machdep.cpu.leaf7_features: SMEP AVX2 BMI2 AVX512F\n"
)


def test_parse_sysctl_apple_silicon():
    info = parse_sysctl(APPLE_SILICON, "arm64")
    assert info == CpuInfo(arch="arm64", system="Darwin", neon=True)


def test_parse_sysctl_intel_mac():
    info = parse_sysctl(INTEL_MAC, "x86_64")
    assert info == CpuInfo(arch="x86_64", system="Darwin", avx2=True, avx512f=True)


X86_LINUX = "processor\t: 0\nflags\t\t: fpu sse sse2 avx avx2 fma\n"
ARM_LINUX = "processor\t: 0\nFeatures\t: fp asimd evtstrm aes\n"


def test_parse_proc_cpuinfo_x86_64():
    info = parse_proc_cpuinfo(X86_LINUX, "AMD64")
    assert info == CpuInfo(arch="x86_64", system="Linux", avx2=True)


def test_parse_proc_cpuinfo_aarch64():
    info = parse_proc_cpuinfo(ARM_LINUX, "aarch64")
    assert info == CpuInfo(arch="arm64", system="Linux", neon=True)


def test_cpuinfo_is_frozen():
    info = CpuInfo(arch="arm64", system="Linux")
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.neon = True  # type: ignore[misc]
