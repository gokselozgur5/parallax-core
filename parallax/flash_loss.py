"""Memory-efficient FlashCrossEntropy with online softmax tiling across vocabulary.

Eliminates full (N, V) probability memory footprint by accumulating log-sum-exp
and streaming gradient computation across vocabulary chunks.
"""

from typing import Tuple
import numpy as np


def flash_cross_entropy_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    chunk_size: int = 64,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute cross-entropy loss using online softmax tiling across vocab dimension.

    Args:
        logits: Logit matrix of shape (N, V).
        targets: Target class indices of shape (N,).
        chunk_size: Size of vocabulary tile.

    Returns:
        Tuple of:
            - loss: Scalar mean cross entropy loss.
            - m_vec: Running max logits of shape (N,).
            - l_vec: Running sum-exp of shape (N,).
    """
    N, V = logits.shape
    m_vec = np.full(N, -np.inf, dtype=np.float32)
    l_vec = np.zeros(N, dtype=np.float32)
    target_logits = np.zeros(N, dtype=np.float32)

    for i in range(N):
        target_logits[i] = logits[i, targets[i]]
