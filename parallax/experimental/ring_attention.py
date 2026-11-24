"""Ring Attention blockwise parallel attention kernel."""

from typing import Tuple
import numpy as np


class RingAttentionContext:
    """Manages blockwise state for Ring Attention simulation across workers."""

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = int(block_size)

    def forward(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        causal: bool = True,
    ) -> np.ndarray:
        """Run ring attention using this context's block_size."""
        return ring_attention_forward(q, k, v, block_size=self.block_size, causal=causal)


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
    both_neginf = np.isneginf(new_m)
    safe_new_m = np.where(both_neginf, 0.0, new_m)
    safe_acc_m = np.where(np.isneginf(acc_m), 0.0, acc_m)
    safe_block_m = np.where(np.isneginf(block_m), 0.0, block_m)
    alpha = np.where(np.isneginf(acc_m), 0.0, np.exp(safe_acc_m - safe_new_m))
    beta = np.where(np.isneginf(block_m), 0.0, np.exp(safe_block_m - safe_new_m))
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
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(
            f"q, k, v must be 2D arrays, got shapes {q.shape}, {k.shape}, {v.shape}"
        )
    Tq, dim_q = q.shape
    Tk, dim_k = k.shape
    Tv, dim_v = v.shape
    if dim_q != dim_k or Tk != Tv or dim_k != dim_v:
        raise ValueError(
            f"Dimension mismatch: q {q.shape}, k {k.shape}, v {v.shape}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if Tq == 0:
        return np.zeros((0, dim_q), dtype=q.dtype)
    if Tk == 0:
        raise ValueError("k and v sequence length Tk must be at least 1")

    scale = 1.0 / np.sqrt(dim_q)
    out = np.zeros((Tq, dim_q), dtype=q.dtype)
    num_q_blocks = (Tq + block_size - 1) // block_size
    num_k_blocks = (Tk + block_size - 1) // block_size

    for i in range(num_q_blocks):
        q_start, q_end = i * block_size, min((i + 1) * block_size, Tq)
        qi = q[q_start:q_end]
        acc_o = np.zeros((q_end - q_start, dim_q))
        acc_m = np.full(q_end - q_start, -np.inf)
        acc_l = np.zeros(q_end - q_start)

        for j in range(num_k_blocks):
            if causal and j > i:
                continue
            k_start, k_end = j * block_size, min((j + 1) * block_size, Tk)
            kj, vj = k[k_start:k_end], v[k_start:k_end]
            scores = np.matmul(qi, kj.T) * scale
            if causal and i == j:
                qi_idx = np.arange(q_start, q_end)[:, None]
                kj_idx = np.arange(k_start, k_end)[None, :]
                scores = np.where(qi_idx >= kj_idx, scores, -np.inf)
            block_m = np.max(scores, axis=-1)
            all_masked = np.isneginf(block_m)
            safe_block_m = np.where(all_masked, 0.0, block_m)
            exp_s = np.exp(scores - safe_block_m[:, None])
            exp_s = np.where(all_masked[:, None], 0.0, exp_s)
            block_l = np.sum(exp_s, axis=-1)
            block_o = np.matmul(exp_s, vj) / np.maximum(block_l[:, None], 1e-12)
            acc_o, acc_m, acc_l = ring_flash_update(
                acc_o, acc_m, acc_l, block_o, block_m, block_l
            )
        out[q_start:q_end] = acc_o
    return out
