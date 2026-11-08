"""Top-A (threshold = alpha * p_max^2) dynamic truncation sampler."""

from typing import Tuple
import numpy as np


class TopASampler:
    """Dynamically truncates vocabulary tail when confidence is high."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = float(alpha)
