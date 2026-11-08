"""ALiBi (Attention with Linear Biases) positional bias kernel."""

from typing import Optional
import numpy as np


def get_alibi_slopes(num_heads: int) -> np.ndarray:
    """Compute geometric slopes for ALiBi multi-head attention."""
    def get_slopes_power_of_2(n: int) -> list:
        start = 2.0 ** (-(2.0 ** -(np.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if (num_heads & (num_heads - 1)) == 0:
        return np.array(get_slopes_power_of_2(num_heads), dtype=np.float32)
    # Non-power-of-two fallback
    closest_power_of_2 = 2 ** int(np.floor(np.log2(num_heads)))
    slopes = get_slopes_power_of_2(closest_power_of_2)
    extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
    return np.array(slopes + extra_slopes, dtype=np.float32)

def build_alibi_bias_matrix(num_heads: int, seq_len: int) -> np.ndarray:
    """Construct (num_heads, seq_len, seq_len) linear distance bias matrix."""
    slopes = get_alibi_slopes(num_heads)[:, None, None]
    positions = np.arange(seq_len)
    relative_dist = positions[None, :] - positions[:, None]
    # For causal attention, relative_dist is negative or zero (j - i)
    return slopes * relative_dist[None, :, :]

def alibi_causal_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Multi-head causal attention with ALiBi positional bias added to scores."""
    num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, k.swapaxes(-1, -2)) * scale
    bias = build_alibi_bias_matrix(num_heads, seq_len)
    scores = scores + bias
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(causal_mask == 1, scores, -1e9)
    exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return np.matmul(weights, v)
