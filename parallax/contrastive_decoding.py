"""Contrastive Decoding (expert vs amateur logit subtraction)."""

from typing import Optional
import numpy as np


class ContrastiveDecoder:
    """Computes contrastive logits between expert and amateur model outputs."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
