"""
Neural network modules and foundational layers.

Provides the Module abstraction, Parameter wrappers, and core layers
including Linear projections, LayerNorm, and RMSNorm.
"""

from typing import List, Optional
import numpy as np
from parallax.tensor import Tensor, _accumulate_grad


class Parameter(Tensor):
    """A Tensor that represents a learnable parameter in a Module."""

    def __init__(self, data: np.ndarray, label: str = ""):
        super().__init__(data, requires_grad=True, label=label)
        self.requires_grad = True


class Module:
    """Base class for all neural network modules."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        """Returns all unique learnable parameters belonging to this module and submodules."""
        params: List[Parameter] = []
        seen_ids = set()

        def collect(m: Module) -> None:
            for name, value in m.__dict__.items():
                if isinstance(value, Parameter):
                    if id(value) not in seen_ids:
                        seen_ids.add(id(value))
                        params.append(value)
                elif isinstance(value, Module):
                    collect(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Module):
                            collect(item)
                        elif isinstance(item, Parameter):
                            if id(item) not in seen_ids:
                                seen_ids.add(id(item))
                                params.append(item)

        collect(self)
        return params

    def zero_grad(self) -> None:
        """Resets gradients of all parameters to zero."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Sets the module in training mode (recursively for all submodules)."""
        self.training = mode
        for child in self.__dict__.values():
            if isinstance(child, Module):
                child.train(mode)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Module):
                        item.train(mode)
        return self

    def eval(self) -> "Module":
        """Sets the module in evaluation mode."""
        return self.train(False)


class Linear(Module):
    """Linear transformation: y = x @ W^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier / Glorot uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        weight_data = np.random.uniform(-limit, limit, (out_features, in_features))
        self.weight = Parameter(weight_data, label="Linear.weight")

        if bias:
            bias_data = np.zeros(out_features)
            self.bias: Optional[Parameter] = Parameter(bias_data, label="Linear.bias")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., in_features], weight: [out_features, in_features]
        # x @ weight^T -> [..., out_features]
        w_t = self.weight.transpose(1, 0)
        out = x @ w_t
        if self.bias is not None:
            out = out + self.bias
        return out


class RMSNorm(Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Standard normalization used in LLaMA, Mistral, and modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(dim), label="RMSNorm.weight")

    def forward(self, x: Tensor) -> Tensor:
        # RMS = sqrt(mean(x^2, axis=-1, keepdims=True) + eps)
        # norm_x = x / RMS * weight
        x_data = x.data
        rms = np.sqrt(np.mean(x_data**2, axis=-1, keepdims=True) + self.eps)
        norm_x = x_data / rms
        out_data = norm_x * self.weight.data

        out = Tensor(out_data, (x, self.weight), "RMSNorm")

        if not out.requires_grad:
            return out

        def _backward() -> None:
            if out.grad is None:
                return
            if self.weight.requires_grad:
                # dL/dWeight = sum(dL/dOut * norm_x)
                grad_w = out.grad * norm_x
                _accumulate_grad(self.weight, grad_w)

            if x.requires_grad:
                # Analytical gradient of RMSNorm w.r.t x:
                # dL/dx = (g * w / RMS) - (x / (d * RMS^3)) * sum(g * w * x)
                g_w = out.grad * self.weight.data
                d = x_data.shape[-1]
                sum_gwx = np.sum(g_w * x_data, axis=-1, keepdims=True)
                dx = (g_w / rms) - (x_data / (d * (rms**3))) * sum_gwx
                _accumulate_grad(x, dx)

        out._backward = _backward
        return out


class LayerNorm(Module):
    """Layer Normalization (Ba, Kiros & Hinton, 2016)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(dim), label="LayerNorm.weight")
        self.bias = Parameter(np.zeros(dim), label="LayerNorm.bias")

    def forward(self, x: Tensor) -> Tensor:
        x_data = x.data
        mean = np.mean(x_data, axis=-1, keepdims=True)
        var = np.var(x_data, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x_data - mean) * std_inv
        out_data = x_hat * self.weight.data + self.bias.data

        out = Tensor(out_data, (x, self.weight, self.bias), "LayerNorm")

        if not out.requires_grad:
            return out

        def _backward() -> None:
            if out.grad is None:
                return
            if self.weight.requires_grad:
                _accumulate_grad(self.weight, out.grad * x_hat)
            if self.bias.requires_grad:
                _accumulate_grad(self.bias, out.grad)

            if x.requires_grad:
                d = x_data.shape[-1]
                dout = out.grad * self.weight.data
                dx_hat = dout
                dvar = np.sum(dx_hat * (x_data - mean) * -0.5 * (std_inv**3), axis=-1, keepdims=True)
                dmean = np.sum(dx_hat * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x_data - mean), axis=-1, keepdims=True)
                dx = dx_hat * std_inv + dvar * 2.0 * (x_data - mean) / d + dmean / d
                _accumulate_grad(x, dx)

        out._backward = _backward
        return out
