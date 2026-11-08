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
    alpha = np.exp(acc_m - new_m)
    beta = np.exp(block_m - new_m)
    new_l = acc_l * alpha + block_l * beta
    new_o = (acc_o * alpha[..., None] + block_o * beta[..., None]) / np.maximum(new_l[..., None], 1e-12)
    return new_o, new_m, new_l
