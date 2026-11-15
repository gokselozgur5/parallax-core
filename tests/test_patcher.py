import pytest
from parallax.backends.linear import DispatchedLinear
from parallax.nn import Linear, Module
from parallax.nn.patcher import disable_native, enable_native


def test_enable_and_disable_single():
    lin = Linear(4, 2)
    native = enable_native(lin)
    assert isinstance(native, DispatchedLinear)
    restored = disable_native(native)
    assert type(restored) is Linear


class MLP(Module):
    def __init__(self) -> None:
        self.fc1 = Linear(4, 8)
        self.layers = [Linear(8, 8), Linear(8, 2)]


def test_enable_and_disable_container() -> None:
    model = MLP()
    enable_native(model)
    assert isinstance(model.fc1, DispatchedLinear)
    assert all(isinstance(ly, DispatchedLinear) for ly in model.layers)
    disable_native(model)
    assert type(model.fc1) is Linear
    assert all(type(ly) is Linear for ly in model.layers)


def test_monkey_patch_linear_reversibly() -> None:
    from parallax.nn.patcher import is_patched, patch_linear, unpatch_linear
    assert not is_patched()
    patch_linear()
    assert is_patched()
    lin = Linear(4, 2)
    assert isinstance(lin, DispatchedLinear)
    unpatch_linear()
    assert not is_patched()
    lin2 = Linear(4, 2)
    assert type(lin2) is Linear
