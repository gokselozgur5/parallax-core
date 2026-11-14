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
