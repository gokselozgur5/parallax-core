"""Contrastive Decoding (expert vs amateur logit subtraction)."""

from typing import Optional
import numpy as np


class ContrastiveDecoder:
    """Computes contrastive logits between expert and amateur model outputs."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    def compute_contrastive_logits(
        self,
        expert_logits: np.ndarray,
        amateur_logits: np.ndarray,
    ) -> np.ndarray:
        """Subtract scaled amateur logits from expert logits with plausibility mask."""
        # Plausibility threshold: keep tokens with expert prob >= beta * max_expert_prob
        exp_expert = np.exp(expert_logits - np.max(expert_logits, axis=-1, keepdims=True))
        expert_probs = exp_expert / np.sum(exp_expert, axis=-1, keepdims=True)
        max_prob = np.max(expert_probs, axis=-1, keepdims=True)
        plausible_mask = expert_probs >= (self.beta * max_prob)

        cd_logits = (1.0 + self.alpha) * expert_logits - self.alpha * amateur_logits
        return np.where(plausible_mask, cd_logits, -1e9)

    def sample_next_token(
        self,
        expert_logits: np.ndarray,
        amateur_logits: np.ndarray,
        temperature: float = 1.0,
    ) -> int:
        """Sample next token using contrastive logits distribution."""
        cd_logits = self.compute_contrastive_logits(expert_logits, amateur_logits)
        scaled_logits = cd_logits / max(temperature, 1e-5)
        exp_l = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_l / np.sum(exp_l)
        return int(np.random.choice(len(probs), p=probs))
