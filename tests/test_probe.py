import dataclasses
import pytest
from parallax.core.probe import CpuFeatures, CpuProbe


def test_cpu_features_defaults():
    info = CpuFeatures(arch="x86_64")
    assert info.arch == "x86_64"
    assert not info.neon
    assert not info.avx2
    assert not info.avx512


def test_cpu_features_frozen():
    info = CpuFeatures(arch="arm64", neon=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.neon = False  # type: ignore[misc]


def test_cpu_probe_alias():
    assert CpuProbe is CpuFeatures


from parallax.core.probe import normalize_arch


def test_normalize_arch():
    assert normalize_arch("aarch64") == "arm64"
    assert normalize_arch("ARM64") == "arm64"
    assert normalize_arch("AMD64") == "x86_64"
    assert normalize_arch("x86_64") == "x86_64"
    assert normalize_arch("unknown_arch") == "unknown_arch"


def test_cpu_features_helpers():
    info = CpuFeatures(arch="x86_64", avx2=True, avx512=True)
    assert info.has_avx2
    assert info.has_avx512
    assert info.avx512f
    assert not info.has_neon
    assert info.has_feature("avx2")
    assert info.has_feature("AVX-512")
    assert not info.has_feature("neon")


from parallax.core.probe import parse_proc_cpuinfo

X86_LINUX = "processor\t: 0\nflags\t\t: fpu sse sse2 avx avx2 avx512f\n"
ARM_LINUX = "processor\t: 0\nFeatures\t: fp asimd evtstrm aes\n"


def test_parse_proc_cpuinfo_x86_64():
    info = parse_proc_cpuinfo(X86_LINUX, "AMD64")
    assert info.arch == "x86_64"
    assert info.system == "Linux"
    assert info.avx2
    assert info.avx512
    assert not info.neon


def test_parse_proc_cpuinfo_arm64():
    info = parse_proc_cpuinfo(ARM_LINUX, "aarch64")
    assert info.arch == "arm64"
    assert info.system == "Linux"
    assert info.neon
    assert not info.avx2


from parallax.core.probe import parse_sysctl

APPLE_SYSCTL = "hw.optional.neon: 1\nhw.optional.AdvSIMD: 1\nmachdep.cpu.brand_string: Apple M3\n"
INTEL_SYSCTL = "machdep.cpu.features: FPU SSE SSE2\nmachdep.cpu.leaf7_features: SMEP AVX2 AVX512F\n"


def test_parse_sysctl_apple():
    info = parse_sysctl(APPLE_SYSCTL, "arm64")
    assert info.arch == "arm64"
    assert info.system == "Darwin"
    assert info.neon
    assert not info.avx2


def test_parse_sysctl_intel():
    info = parse_sysctl(INTEL_SYSCTL, "x86_64")
    assert info.arch == "x86_64"
    assert info.system == "Darwin"
    assert info.avx2
    assert info.avx512
    assert not info.neon


import platform
from parallax.core.probe import detect_cpu, probe_cpu


def test_probe_cpu_live():
    info = probe_cpu()
    assert isinstance(info, CpuFeatures)
    assert detect_cpu() == info
    assert info.arch in ("x86_64", "arm64", platform.machine().lower())


def test_probe_cpu_fallback(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "UnknownOS")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    info = probe_cpu()
    assert info.arch == "x86_64"
    assert info.system == "UnknownOS"
    assert not info.neon
    assert not info.avx2
    assert not info.avx512
