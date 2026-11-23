import pytest
from parallax.backends.linear import DispatchedLinear
from parallax.nn import Linear, Module
from parallax.nn.patcher import disable_native, enable_native


@pytest.fixture(autouse=True)
def setup_backend():
    from parallax.backends import registry
    from parallax.backends.numpy_backend import NumpyBackend
    registry.clear()
    registry.register("numpy", NumpyBackend())
    yield
    registry.clear()


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
    import parallax.nn as nn
    assert not is_patched()
    patch_linear()
    assert is_patched()
    lin = nn.Linear(4, 2)
    assert isinstance(lin, DispatchedLinear)
    unpatch_linear()
    assert not is_patched()
    lin2 = nn.Linear(4, 2)
    assert type(lin2) is Linear


def test_param_identity_and_parity() -> None:
    import numpy as np
    from parallax.tensor import Tensor

    lin = Linear(4, 3, bias=True)
    orig_w, orig_b = lin.weight, lin.bias
    x = Tensor(np.random.randn(2, 4))
    y_orig = lin(x)
    native = enable_native(lin)
    assert native.weight is orig_w
    assert native.bias is orig_b
    assert np.allclose(native(x).data, y_orig.data, atol=1e-6)
    restored = disable_native(native)
    assert restored.weight is orig_w
    assert restored.bias is orig_b
    assert np.allclose(restored(x).data, y_orig.data, atol=1e-6)


def test_patch_linear_reaches_models_built_afterwards():
    import numpy as np
    import parallax.nn as nn
    from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
    from parallax.nn.patcher import patch_linear, unpatch_linear

    np.random.seed(0)
    plain_linear = nn.Linear
    patch_linear()
    try:
        model = ParallaxLM(ParallaxLMConfig(vocab_size=8, embed_dim=8, num_heads=2, num_layers=1))
    finally:
        unpatch_linear()
    assert type(model.lm_head) is not plain_linear, type(model.lm_head)
    assert type(model.lm_head).__name__ == "DispatchedLinear", type(model.lm_head)
