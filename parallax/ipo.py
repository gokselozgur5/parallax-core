"""Identity Preference Optimization (IPO) direct alignment kernel."""

from typing import Tuple
import numpy as np


class IPOConfig:
    """Configuration container for Identity Preference Optimization."""

    def __init__(self, tau: float = 0.1) -> None:
        self.tau = float(tau)
        self.target_margin = 1.0 / (2.0 * self.tau)
