"""
Low-Rank Adaptation (LoRA: Hu et al., 2021) implemented from first principles.

Parameter-Efficient Fine-Tuning (PEFT) module that freezes pre-trained linear
weights W_0 and injects trainable rank decomposition matrices:
    W = W_0 + (alpha / r) * (B @ A)
where A ~ N(0, sigma^2) and B = 0, so initially delta_W = 0.
"""

from typing import Any, List, Optional, Tuple
import numpy as np

from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor


class LoRALinear(Module):
    """Linear layer wrapped with Low-Rank Adaptation (LoRA).

    Forward pass:
        h = x @ W_0^T + (alpha / r) * (x @ A^T @ B^T) + bias
    """

    def __init__(
        self,
        base_layer: Linear,
        r: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Freeze base parameters
        self.base_layer.weight.requires_grad = False
        self.base_layer.weight.grad = None
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            self.base_layer.bias.grad = None

        self.lora_A: Optional[Parameter]
        self.lora_B: Optional[Parameter]
        if r > 0:
            # A initialized with Gaussian N(0, 1/r), B initialized to 0
            limit_a = 1.0 / np.sqrt(self.in_features)
            a_data = np.random.normal(0.0, limit_a, (r, self.in_features))
            b_data = np.zeros((self.out_features, r), dtype=np.float64)

            self.lora_A = Parameter(a_data, label=f"LoRA_A_{self.in_features}x{r}")
            self.lora_B = Parameter(b_data, label=f"LoRA_B_{r}x{self.out_features}")
        else:
            self.lora_A = None
            self.lora_B = None

        self.merged = False
        self.training = True

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: Tensor) -> Tensor:
        # Base forward pass
        base_out = self.base_layer(x)

        if self.r == 0 or self.merged or self.lora_A is None or self.lora_B is None:
            return base_out

        # LoRA path: x: [..., in_features]
        # Inverted dropout on x during training if lora_dropout > 0.0
        lora_input = x
        if self.training and self.lora_dropout > 0.0:
            mask = (np.random.rand(*x.shape) >= self.lora_dropout) / (1.0 - self.lora_dropout)
            lora_input = x * Tensor(mask)

        # x @ A^T -> [..., r]
        # (x @ A^T) @ B^T -> [..., out_features]
        lora_h = lora_input @ self.lora_A.transpose(1, 0)
        lora_out = lora_h @ self.lora_B.transpose(1, 0)

        scaled_lora = lora_out * self.scaling
        return base_out + scaled_lora

    def merge_weights(self) -> None:
        """Merges delta_W = (alpha / r) * (B @ A) directly into base_layer.weight."""
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            # B: [out_features, r], A: [r, in_features] -> delta_W: [out_features, in_features]
            delta_w = self.scaling * (self.lora_B.data @ self.lora_A.data)
            self.base_layer.weight.data += delta_w
            self.merged = True

    def unmerge_weights(self) -> None:
        """Removes delta_W from base_layer.weight."""
        if self.r > 0 and self.merged and self.lora_A is not None and self.lora_B is not None:
            delta_w = self.scaling * (self.lora_B.data @ self.lora_A.data)
            self.base_layer.weight.data -= delta_w
            self.merged = False

    def unwrap(self) -> Linear:
        """Returns the underlying base linear layer."""
        return self.base_layer

    def merge_and_unload(self) -> Linear:
        """Merges LoRA weights into base layer and returns the base layer."""
        if self.r > 0 and not self.merged and self.lora_A is not None and self.lora_B is not None:
            # Add (lora_B.weight @ lora_A.weight) * scaling into base_layer.weight.data
            b_w = self.lora_B.weight if hasattr(self.lora_B, "weight") else self.lora_B
            a_w = self.lora_A.weight if hasattr(self.lora_A, "weight") else self.lora_A
            b_data = getattr(b_w, "data", b_w)
            a_data = getattr(a_w, "data", a_w)
            delta_w = (b_data @ a_data) * self.scaling
            self.base_layer.weight.data += delta_w
            self.merged = True
        return self.base_layer


def apply_lora_to_model(
    model: Module,
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "out_proj"),
    r: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
) -> List[LoRALinear]:
    """Inspects a model hierarchy, freezes all non-LoRA parameters, and replaces

    matching Linear layers with LoRALinear wrappers.

    Returns:
        List of created LoRALinear instances.
    """
    # 1. Freeze base model parameters, skipping existing LoRA adapter parameters
    for p in model.parameters():
        if getattr(p, "label", None) and "lora_" in p.label.lower():
            continue
        p.requires_grad = False
        p.grad = None

    lora_layers: List[LoRALinear] = []

    def recursive_apply(module: Module) -> None:
        for attr_name, child in list(module.__dict__.items()):
            if isinstance(child, Linear) and any(t in attr_name for t in target_modules):
                # Replace with LoRALinear
                lora_layer = LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                setattr(module, attr_name, lora_layer)
                lora_layers.append(lora_layer)
            elif isinstance(child, Module):
                recursive_apply(child)
            elif isinstance(child, (list, tuple)):
                new_items: List[Any] = []
                tuple_changed = False
                for idx, item in enumerate(child):
                    item_key = f"{attr_name}.{idx}"
                    if isinstance(item, Linear) and (any(t in attr_name for t in target_modules) or any(t in str(idx) for t in target_modules) or any(t in item_key for t in target_modules)):
                        lora_layer = LoRALinear(item, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                        new_items.append(lora_layer)
                        lora_layers.append(lora_layer)
                        tuple_changed = True
                    elif isinstance(item, Module):
                        recursive_apply(item)
                        new_items.append(item)
                    else:
                        new_items.append(item)
                if isinstance(child, list):
                    child[:] = new_items
                elif tuple_changed:
                    setattr(module, attr_name, tuple(new_items))
            elif isinstance(child, dict):
                for key, item in list(child.items()):
                    if isinstance(item, Linear) and (any(t in attr_name for t in target_modules) or any(t in key for t in target_modules)):
                        lora_layer = LoRALinear(item, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                        child[key] = lora_layer
                        lora_layers.append(lora_layer)
                    elif isinstance(item, Module):
                        recursive_apply(item)

    recursive_apply(model)
    return lora_layers
