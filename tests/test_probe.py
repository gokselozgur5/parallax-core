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
