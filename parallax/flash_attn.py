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

