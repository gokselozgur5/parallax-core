"""Ring Attention blockwise parallel attention kernel."""

from typing import Optional, Tuple
import numpy as np


class RingAttentionContext:
    """Manages blockwise state for Ring Attention simulation across workers."""

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = int(block_size)

def ring_flash_update(
    acc_o: np.ndarray,
    acc_m: np.ndarray,
    acc_l: np.ndarray,
    block_o: np.ndarray,
    block_m: np.ndarray,
    block_l: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge partial block attention outputs via online softmax rescaling."""
    new_m = np.maximum(acc_m, block_m)
    alpha = np.where(np.isneginf(acc_m), 0.0, np.exp(acc_m - new_m))
    beta = np.exp(block_m - new_m)
    new_l = acc_l * alpha + block_l * beta
    new_o = (acc_o * (acc_l * alpha)[..., None] + block_o * (block_l * beta)[..., None]) / np.maximum(new_l[..., None], 1e-12)
    return new_o, new_m, new_l

def ring_attention_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    block_size: int = 16,
    causal: bool = True,
) -> np.ndarray:
    """Compute blockwise ring attention over sequence length in chunks."""
    seq_len, dim = q.shape
    scale = 1.0 / np.sqrt(dim)
    out = np.zeros_like(q)
    num_blocks = (seq_len + block_size - 1) // block_size

    for i in range(num_blocks):
        q_start, q_end = i * block_size, min((i + 1) * block_size, seq_len)
        qi = q[q_start:q_end]
        acc_o = np.zeros((q_end - q_start, dim))
        acc_m = np.full(q_end - q_start, -np.inf)
        acc_l = np.zeros(q_end - q_start)

        for j in range(num_blocks):
            if causal and j > i:
                continue
            k_start, k_end = j * block_size, min((j + 1) * block_size, seq_len)
            kj, vj = k[k_start:k_end], v[k_start:k_end]
            scores = np.matmul(qi, kj.T) * scale
            if causal and i == j:
                qi_idx = np.arange(q_start, q_end)[:, None]
                kj_idx = np.arange(k_start, k_end)[None, :]
                scores = np.where(qi_idx >= kj_idx, scores, -1e9)
            block_m = np.max(scores, axis=-1)
            exp_s = np.exp(scores - block_m[:, None])
            block_l = np.sum(exp_s, axis=-1)
            block_o = np.matmul(exp_s, vj)
            new_m = np.maximum(acc_m, block_m)
            alpha = np.where(np.isneginf(acc_m), 0.0, np.exp(acc_m - new_m))
            beta = np.exp(block_m - new_m)
            acc_l = acc_l * alpha + block_l * beta
            acc_o = acc_o * alpha[..., None] + block_o * beta[..., None]
            acc_m = new_m
        out[q_start:q_end] = acc_o / acc_l[..., None]
    return out
