import numpy as np
import pytest
from parallax.nn import LayerNorm, Linear, Parameter, RMSNorm
from parallax.tensor import Tensor, no_grad


def test_parameter_under_no_grad():
    with no_grad():
        p = Parameter(np.ones(3))
        assert p.requires_grad is True


def test_linear_forward_backward():
    np.random.seed(42)
    layer = Linear(in_features=4, out_features=3, bias=True)
    x = Tensor(np.random.randn(2, 4))
    y = layer(x)

    assert y.shape == (2, 3)

    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == (3, 4)
    assert layer.bias.grad is not None
    assert layer.bias.grad.shape == (3,)
    assert x.grad.shape == (2, 4)


def test_rmsnorm_forward_backward():
    np.random.seed(1337)
    norm = RMSNorm(dim=8)
    x = Tensor(np.random.randn(3, 4, 8))
    out = norm(x)

    assert out.shape == (3, 4, 8)

    loss = out.sum()
    loss.backward()

    assert norm.weight.grad.shape == (8,)
    assert x.grad.shape == (3, 4, 8)


def test_layernorm_forward_backward():
    np.random.seed(999)
    norm = LayerNorm(dim=6)
    x = Tensor(np.random.randn(2, 5, 6))
    out = norm(x)

    assert out.shape == (2, 5, 6)

    loss = out.sum()
    loss.backward()

    assert norm.weight.grad.shape == (6,)
    assert norm.bias.grad.shape == (6,)
    assert x.grad.shape == (2, 5, 6)


def test_unused_parameter_grad_is_none_after_backward():
    np.random.seed(42)
    layer = Linear(4, 3, bias=True)
    assert layer.weight.grad is None
    assert layer.bias.grad is None

    x = Tensor(np.random.randn(2, 4))
    # Multiply x with weight only, leaving bias unused
    y = x @ layer.weight.transpose(1, 0)
    loss = y.sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.bias.grad is None


def test_module_train_eval_mode():
    from parallax.lora import LoRALinear
    base = Linear(8, 4)
    lora = LoRALinear(base, r=2, lora_alpha=4.0, lora_dropout=0.5)
    lora.lora_B.data.fill(1.0)

    lora.eval()
    assert lora.training is False
    x = Tensor(np.ones((2, 8)))
    out1 = lora(x).data
    out2 = lora(x).data
    assert np.allclose(out1, out2)

    lora.train()
    assert lora.training is True
    out3 = lora(x).data
    out4 = lora(x).data
    assert not np.allclose(out3, out4)


def test_no_grad_skips_backward_closure_in_layers():
    norm = RMSNorm(dim=4)
    ln = LayerNorm(dim=4)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)

    with no_grad():
        out_rms = norm(x)
        out_ln = ln(x)

    assert out_rms.requires_grad is False
    assert out_rms._backward.__closure__ is None
    assert out_ln.requires_grad is False
    assert out_ln._backward.__closure__ is None


def test_container_submodules_parameters_and_train():
    from parallax.nn import Module

    class ContainerModel(Module):
        def __init__(self):
            super().__init__()
            self.tup = (Linear(4, 4), Linear(4, 2))
            self.mapping = {"head": Linear(2, 1), "extra": Parameter(np.zeros((1,)))}

    model = ContainerModel()
    params = model.parameters()
    # 2 linears in tup (4 params) + 1 linear in mapping (2 params) + 1 Parameter = 7 params
    assert len(params) == 7

    named = dict(model.named_parameters())
    assert "tup.0.weight" in named
    assert "tup.1.bias" in named
    assert "mapping.head.weight" in named
    assert "mapping.extra" in named

    model.eval()
    assert model.tup[0].training is False
    assert model.mapping["head"].training is False
    model.train()
    assert model.tup[0].training is True
    assert model.mapping["head"].training is True


