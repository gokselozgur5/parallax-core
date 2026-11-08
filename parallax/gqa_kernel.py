"""Grouped-Query Attention (GQA) head expansion and KV cache compression.

Reference: Ainslie et al., 2023 - GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints.
Interpolates between Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
by grouping query heads to share a smaller number of key/value heads.
"""

from typing import Tuple
import numpy as np


def repeat_kv_heads(x: np.ndarray, num_groups: int) -> np.ndarray:
    """Repeat key/value heads num_groups times along the head axis.

    Args:
        x: Key or Value tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        num_groups: Number of times each KV head is shared across query heads.

    Returns:
        Expanded tensor of shape (batch, num_kv_heads * num_groups, seq_len, head_dim).
    """
    if num_groups == 1:
        return x
    b, h_kv, s, d = x.shape
    # Expand along new axis and broadcast
    expanded = np.repeat(x, repeats=num_groups, axis=1)
    return expanded



def reduce_gqa_kv_gradients(grad_repeated: np.ndarray, num_groups: int) -> np.ndarray:
    """Accumulate backpropagated gradients across repeated query head groups back to KV heads.

    Args:
        grad_repeated: Gradient tensor of shape (batch, num_kv_heads * num_groups, seq_len, head_dim).
        num_groups: Group multiplier.

    Returns:
        Reduced gradient tensor of shape (batch, num_kv_heads, seq_len, head_dim).
    """
    if num_groups == 1:
        return grad_repeated
    b, total_heads, s, d = grad_repeated.shape
    num_kv_heads = total_heads // num_groups

    # Reshape to (b, num_kv_heads, num_groups, s, d) and sum over group axis
    reshaped = grad_repeated.reshape(b, num_kv_heads, num_groups, s, d)
    return np.sum(reshaped, axis=2)
