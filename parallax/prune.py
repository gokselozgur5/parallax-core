"""
Zero-dependency Neural Network Pruning and 2:4 Structured Sparsity Engine.

Implements magnitude-based unstructured pruning and NVIDIA Sparse Tensor Core
compliant 2:4 structured sparsity patterns from first principles using NumPy.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from parallax.models.gpt import ParallaxLM
from parallax.nn import Module


def prune_unstructured_magnitude(
    weights: np.ndarray,
    sparsity: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies unstructured magnitude pruning to a weight array."""
    if not (0.0 <= sparsity <= 1.0):
        raise ValueError("sparsity must be between 0.0 and 1.0")
    if sparsity == 0.0:
        return weights.copy(), np.ones_like(weights, dtype=bool)
    if sparsity == 1.0:
        return np.zeros_like(weights), np.zeros_like(weights, dtype=bool)

    abs_vals = np.abs(weights).flatten()
    n = len(abs_vals)
    k = int(np.round(sparsity * n))
    k = min(max(k, 1), n - 1)

    pruned_indices = np.argpartition(abs_vals, k)[:k]
    mask_flat = np.ones(n, dtype=bool)
    mask_flat[pruned_indices] = False
    mask = mask_flat.reshape(weights.shape)

    pruned = weights * mask
    return pruned, mask


def prune_2_to_4_structured(
    weights: np.ndarray,
    sparsity: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies NVIDIA 2:4 structured sparsity (exactly 2 zeroes out of every 4 elements)."""
    if sparsity is not None and abs(sparsity - 0.5) > 1e-4:
        raise ValueError(
            "2:4 structured pruning strictly requires sparsity=0.5 (2 non-zeros per 4 elements)."
        )
    orig_shape = weights.shape
    total_elements = weights.size
    if total_elements % 4 != 0:
        raise ValueError(
            f"Total elements ({total_elements}) must be divisible by 4 for 2:4 structured sparsity"
        )

    reshaped = weights.reshape(-1, 4)
    abs_vals = np.abs(reshaped)

    # Find the indices of the 2 smallest values in each 4-element block
    smallest_2_indices = np.argsort(abs_vals, axis=-1)[:, :2]

    # Create boolean mask: True for kept, False for pruned
    mask_blocks = np.ones_like(reshaped, dtype=bool)
    np.put_along_axis(mask_blocks, smallest_2_indices, False, axis=-1)

    mask = mask_blocks.reshape(orig_shape)
    pruned = weights * mask
    return pruned, mask


def calculate_model_sparsity(model: Module) -> Dict[str, float]:
    """Computes exact non-zero and total parameter counts and overall sparsity ratio."""
    total_elements = 0
    zero_elements = 0

    for param in model.parameters():
        total_elements += param.data.size
        zero_elements += int(np.sum(param.data == 0.0))

    sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
    return {
        "total_elements": float(total_elements),
        "zero_elements": float(zero_elements),
        "sparsity_ratio": float(sparsity),
    }


def apply_pruning_masks(model: Module) -> None:
    """Re-applies stored pruning masks to parameters (e.g. after optimizer steps)."""
    for param in model.parameters():
        mask = getattr(param, "mask", None)
        if mask is not None:
            param.data = param.data * mask


def create_pruning_step_hook(model: Module):
    """Returns a post-step hook callable that re-zeros masked weights."""
    def hook():
        apply_pruning_masks(model)
    return hook


def prune_model(
    model: ParallaxLM,
    method: str = "2:4",
    sparsity: float = 0.5,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Prunes all target linear layers in a ParallaxLM model."""
    valid_targets = {"attn", "mlp"}
    targets = set(target_modules) if target_modules else valid_targets

    pruned_layers_count = 0
    model_masks: Dict[str, np.ndarray] = {}

    for i, block in enumerate(model.blocks):
        # Attention projection layers
        if "attn" in targets:
            attn_layers = [
                ("q_proj", block.attn.q_proj),
                ("k_proj", block.attn.k_proj),
                ("v_proj", block.attn.v_proj),
                ("out_proj", block.attn.out_proj),
            ]
            for name, layer in attn_layers:
                if method == "2:4":
                    pruned_w, mask = prune_2_to_4_structured(layer.weight.data, sparsity=sparsity)
                elif method == "magnitude":
                    pruned_w, mask = prune_unstructured_magnitude(layer.weight.data, sparsity=sparsity)
                else:
                    raise ValueError(f"Unknown pruning method: {method}")
                layer.weight.data = pruned_w
                setattr(layer.weight, "mask", mask)
                model_masks[f"block.{i}.attn.{name}"] = mask
                pruned_layers_count += 1

        # MLP projection layers
        if "mlp" in targets:
            mlp_layers = []
            mlp: Any = block.mlp
            if hasattr(mlp, "experts"):
                for exp_idx, exp in enumerate(mlp.experts):
                    if hasattr(exp, "w_gate"):
                        mlp_layers.extend([
                            (f"expert.{exp_idx}.w_gate", exp.w_gate),
                            (f"expert.{exp_idx}.w_up", exp.w_up),
                            (f"expert.{exp_idx}.w_down", exp.w_down),
                        ])
                    elif hasattr(exp, "fc1"):
                        mlp_layers.extend([
                            (f"expert.{exp_idx}.fc1", exp.fc1),
                            (f"expert.{exp_idx}.fc2", exp.fc2),
                        ])
            elif hasattr(mlp, "w_gate"):
                mlp_layers.extend([
                    ("w_gate", mlp.w_gate),
                    ("w_up", mlp.w_up),
                    ("w_down", mlp.w_down),
                ])
            elif hasattr(mlp, "fc1"):
                mlp_layers.extend([
                    ("fc1", mlp.fc1),
                    ("fc2", mlp.fc2),
                ])

            for name, layer in mlp_layers:
                if method == "2:4":
                    pruned_w, mask = prune_2_to_4_structured(layer.weight.data, sparsity=sparsity)
                elif method == "magnitude":
                    pruned_w, mask = prune_unstructured_magnitude(layer.weight.data, sparsity=sparsity)
                else:
                    raise ValueError(f"Unknown pruning method: {method}")
                layer.weight.data = pruned_w
                setattr(layer.weight, "mask", mask)
                model_masks[f"block.{i}.mlp.{name}"] = mask
                pruned_layers_count += 1

    stats = calculate_model_sparsity(model)
    stats["pruned_layers_count"] = float(pruned_layers_count)
    setattr(model, "pruning_masks", model_masks)
    return stats
