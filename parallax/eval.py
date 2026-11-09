"""
Perplexity and Language Model Evaluation Suite.

Provides cross-entropy evaluation and sliding-window perplexity (PPL)
computation for autoregressive models without external dependencies.
"""

from typing import List, Optional, Union
import numpy as np
from parallax.models.gpt import ParallaxLM
from parallax.tensor import Tensor


def compute_cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Computes cross-entropy loss from unnormalized logits and integer targets.

    Args:
        logits: Array of shape [N, vocab_size]
        targets: 1D array of shape [N] with ground-truth integer token IDs.

    Returns:
        Mean negative log-likelihood across tokens.
    """
    assert logits.ndim == 2, "logits must be 2D [N, vocab_size]"
    assert targets.ndim == 1, "targets must be 1D [N]"
    assert len(logits) == len(targets), "logits and targets length mismatch"

    # Numerically stable log-softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = (logits - max_logits) - np.log(sum_exp)

    # Pick log prob of true targets
    n_tokens = len(targets)
    target_log_probs = log_probs[np.arange(n_tokens), targets]
    return float(-np.mean(target_log_probs))
