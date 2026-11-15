import pytest
from parallax.backends.linear import DispatchedLinear
from parallax.nn import Linear
from parallax.nn.patcher import disable_native, enable_native


def test_enable_and_disable_single():
    lin = Linear(4, 2)
    native = enable_native(lin)
    assert isinstance(native, DispatchedLinear)
    restored = disable_native(native)
    assert type(restored) is Linear
