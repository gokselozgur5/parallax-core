"""Tests for core backend registry."""

import pytest

from parallax.core.registry import BackendRegistry


def test_register_and_get():
    reg = BackendRegistry()
    engine = object()
    reg.register("cpu", engine)
    assert reg.get("cpu") is engine


def test_register_duplicate_raises():
    reg = BackendRegistry()
    reg.register("cpu", object())
    with pytest.raises(ValueError):
        reg.register("cpu", object())


def test_get_unknown_raises():
    reg = BackendRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_unregister_and_clear():
    reg = BackendRegistry()
    reg.register("cpu", object())
    reg.unregister("cpu")
    with pytest.raises(KeyError):
        reg.unregister("cpu")
    reg.register("cpu", object())
    reg.clear()
    with pytest.raises(KeyError):
        reg.get("cpu")


def test_list_backends_priority_order():
    reg = BackendRegistry()
    reg.register("cpu", object(), priority=0)
    reg.register("cuda", object(), priority=10)
    reg.register("metal", object(), priority=5)
    reg.register("hip", object(), priority=10)
    assert reg.list_backends() == ["cuda", "hip", "metal", "cpu"]


def test_dict_like_access_and_len():
    reg = BackendRegistry()
    eng = object()
    reg.register("cpu", eng, priority=2)
    assert "cpu" in reg
    assert "gpu" not in reg
    assert len(reg) == 1
    assert reg["cpu"] is eng
    del reg["cpu"]
    assert "cpu" not in reg


def test_module_level_helpers():
    from parallax.core.registry import clear, get, list_backends, register, unregister

    clear()
    engine = object()
    register("cpu", engine, priority=1)
    register("gpu", object(), priority=5)
    assert get("cpu") is engine
    assert list_backends() == ["gpu", "cpu"]
    unregister("cpu")
    assert list_backends() == ["gpu"]
    clear()
    assert list_backends() == []


def test_dispatched_linear_default_registry():
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from parallax.nn import DispatchedLinear\n"
        "from parallax.tensor import Tensor\n"
        "print(DispatchedLinear(4, 3)(Tensor(np.ones((2, 4)))).shape)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-300:]
    assert res.stdout.strip() == "(2, 3)"

