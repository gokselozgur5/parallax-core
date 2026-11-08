"""
FlashAttention-2 tiling mathematical primitives and online softmax statistics.
"""

from typing import Tuple
import numpy as np


def update_online_softmax(
    m_old: np.ndarray,
    l_old: np.ndarray,
    O_old: np.ndarray,
    S_block: np.ndarray,
    V_block: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Updates running row-max, sum of exps, and accumulator using FlashAttention-2 online softmax."""
    m_block = np.max(S_block, axis=-1)
    m_new = np.maximum(m_old, m_block)

    P_block = np.exp(S_block - m_new[:, :, :, None])
    alpha = np.exp(m_old - m_new)
    alpha = np.nan_to_num(alpha, nan=0.0)

    l_new = alpha * l_old + np.sum(P_block, axis=-1)
    PV = np.matmul(P_block, V_block)
    O_new = alpha[:, :, :, None] * O_old + PV

    return m_new, l_new, O_new


def reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    is_causal: bool = True,
) -> np.ndarray:
    """Exact full-matrix attention reference implementation."""
    B, H, Tq, d = q.shape
    Tk = k.shape[2]
    scale = 1.0 / np.sqrt(d)
    S = np.matmul(q, k.swapaxes(-2, -1)) * scale
    if is_causal:
        mask = np.arange(Tq)[:, None] < np.arange(Tk)[None, :]
        S = np.where(mask[None, None, :, :], -1e9, S)
    exp_S = np.exp(S - np.max(S, axis=-1, keepdims=True))
    P = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    return np.matmul(P, v)


def flash_attention_forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    is_causal: bool = True,
    block_r: int = 16,
    block_c: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact FlashAttention-2 forward pass using blocked matrix tiling."""
    B, H, Tq, d = q.shape
    Tk = k.shape[2]
    scale = 1.0 / np.sqrt(d)

    o = np.zeros_like(q, dtype=np.float64)
    l = np.zeros((B, H, Tq), dtype=np.float64)
    m = np.full((B, H, Tq), -np.inf, dtype=np.float64)

    Tc = (Tk + block_c - 1) // block_c
    Tr = (Tq + block_r - 1) // block_r

    for j in range(Tc):
        c_start, c_end = j * block_c, min((j + 1) * block_c, Tk)
        K_j, V_j = k[:, :, c_start:c_end, :], v[:, :, c_start:c_end, :]

        for i in range(Tr):
            r_start, r_end = i * block_r, min((i + 1) * block_r, Tq)
            if is_causal and r_end <= c_start:
                continue

            Q_i = q[:, :, r_start:r_end, :]
            S_ij = np.matmul(Q_i, K_j.swapaxes(-2, -1)) * scale

            if is_causal:
                mask = np.arange(r_start, r_end)[:, None] < np.arange(c_start, c_end)[None, :]
                S_ij = np.where(mask[None, None, :, :], -1e9, S_ij)

            m_new, l_new, O_new = update_online_softmax(
                m[:, :, r_start:r_end],
                l[:, :, r_start:r_end],
                o[:, :, r_start:r_end, :],
                S_ij,
                V_j,
            )
            m[:, :, r_start:r_end] = m_new
            l[:, :, r_start:r_end] = l_new
            o[:, :, r_start:r_end, :] = O_new

    safe_l = np.where(l > 0, l, 1.0)[:, :, :, None]
    return o / safe_l, m + np.log(np.maximum(l, 1e-12))


def compute_d_scalar(do: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Computes row-wise scaling factor D_i = rowsum(dO_i * O_i) for FlashAttention backward."""
    return np.sum(do * o, axis=-1)  # Shape: [B, H, Tq]



