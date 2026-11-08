"""Self-Speculative Decoding via layer-skipping and full verification."""

from typing import Callable, List, Tuple
import numpy as np


class SelfSpeculativeEngine:
    """Uses internal layer-skipping early exit to draft tokens without a separate model."""

    def __init__(self, exit_layer: int, total_layers: int) -> None:
        self.exit_layer = int(exit_layer)
        self.total_layers = int(total_layers)
