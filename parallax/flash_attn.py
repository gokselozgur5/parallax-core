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
