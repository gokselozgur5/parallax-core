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
    m_vec = np.full(N, -np.inf, dtype=np.float64)
    l_vec = np.zeros(N, dtype=np.float64)
    target_logits = np.zeros(N, dtype=np.float64)

    for i in range(N):
        target_logits[i] = logits[i, targets[i]]

    num_chunks = int(np.ceil(V / chunk_size))
    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, V)
        chunk = logits[:, c_start:c_end]

        chunk_m = np.max(chunk, axis=1)
        new_m = np.maximum(m_vec, chunk_m)

        # Rescale old accumulator and add new chunk contribution
        l_vec = l_vec * np.exp(m_vec - new_m) + np.sum(np.exp(chunk - new_m[:, np.newaxis]), axis=1)
        m_vec = new_m

    # Log-sum-exp = m + log(l)
    lse = m_vec + np.log(l_vec)
    # Loss = mean(lse - target_logit)
    loss = float(np.mean(lse - target_logits))
    return loss, m_vec, l_vec



def flash_cross_entropy_backward(
    logits: np.ndarray,
    targets: np.ndarray,
    m_vec: np.ndarray,
    l_vec: np.ndarray,
    chunk_size: int = 64,
) -> np.ndarray:
    """Compute gradients of cross-entropy loss with respect to logits in tiles.

    Args:
        logits: Logit matrix of shape (N, V).
        targets: Target class indices of shape (N,).
        m_vec: Global max logits from forward pass (N,).
        l_vec: Global sum-exp accumulators from forward pass (N,).
        chunk_size: Vocab tile size.

    Returns:
        Gradient matrix of shape (N, V).
    """
    N, V = logits.shape
    grad_logits = np.zeros_like(logits)
    num_chunks = int(np.ceil(V / chunk_size))

    for c in range(num_chunks):
        c_start = c * chunk_size
        c_end = min(c_start + chunk_size, V)
        chunk = logits[:, c_start:c_end]

        # P_chunk = exp(chunk - m) / l
        p_chunk = np.exp(chunk - m_vec[:, np.newaxis]) / (l_vec[:, np.newaxis] + 1e-12)
        grad_chunk = p_chunk / N

        # Subtract 1/N for target tokens within this chunk
        for i in range(N):
            tgt = targets[i]
            if c_start <= tgt < c_end:
                grad_chunk[i, tgt - c_start] -= 1.0 / N

        grad_logits[:, c_start:c_end] = grad_chunk

    return grad_logits
