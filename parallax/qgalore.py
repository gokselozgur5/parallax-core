"""Q-GaLore: Quantized Gradient Low-Rank Projection Optimizer.

Reference: Zhang et al., 2024 - Q-GaLore: Quantized GaLore with Low-Rank Projection.
Projects full-rank weight gradients into low-rank SVD subspaces and quantizes
the projected states to 8-bit integers to drastically reduce optimizer memory.
"""

from typing import Tuple, Optional
import numpy as np


class QGaLoreProjector:
    """Computes low-rank subspace basis and handles INT8 quantized gradient projection."""

    def __init__(self, rank: int, update_interval: int = 100):
        self.rank = rank
        self.update_interval = update_interval
        self.step_count = 0
        self.U: Optional[np.ndarray] = None
        self.V_T: Optional[np.ndarray] = None
