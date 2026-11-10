"""Muon (Momentum Orthogonalized by Newton-Schulz) Optimizer.

Reference: Keller Jordan et al., 2024 - Muon: An optimizer for hidden layers in neural networks.
Applies Newton-Schulz iterations to orthogonalize momentum matrices, achieving
superior convergence speed on 2D weight matrices.
"""

from typing import List, Dict, Optional
import numpy as np


def newton_schulz5(G: np.ndarray, steps: int = 5, eps: float = 1e-7) -> np.ndarray:
    """Zeroth-order orthogonalization of matrix G via Newton-Schulz quintic iteration.

    Args:
        G: Gradient or momentum matrix of shape (M, N).
        steps: Number of Newton-Schulz refinement iterations.
        eps: Small epsilon for norm normalization.

    Returns:
        Orthogonalized matrix of shape (M, N).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(np.float64)
    # Spectral radius scaling: divide by Frobenius norm or upper bound
    norm = np.linalg.norm(X) + eps
    X /= norm

    # If M > N, transpose to work with smaller square matrix then transpose back
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True

    I = np.eye(X.shape[0], dtype=np.float64)
    for _ in range(steps):
        A = np.dot(X, X.T)
        B = b * A + c * np.dot(A, A)
        X = a * X + np.dot(B, X)

    if transposed:
        X = X.T

    return X.astype(G.dtype)



class Muon:
    """Muon optimizer with Momentum Orthogonalization for 2D weight matrices."""

    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        # Momentum buffers
        self.m_buffers = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]):
        """Perform a single optimization step."""
        if len(self.params) != len(grads):
            raise ValueError(
                f"Mismatched params ({len(self.params)}) and grads ({len(grads)})"
            )
        for p, g, buf in zip(self.params, grads, self.m_buffers):
            # Update momentum buffer: buf = momentum * buf + g
            buf[:] = self.momentum * buf + g

            if self.nesterov:
                update_matrix = g + self.momentum * buf
            else:
                update_matrix = buf

            # Orthogonalize update matrix using Newton-Schulz
            if update_matrix.ndim == 2:
                ortho_update = newton_schulz5(update_matrix, steps=self.ns_steps)
            else:
                ortho_update = update_matrix / (np.linalg.norm(update_matrix) + 1e-7)

            # Apply parameter update
            p -= self.lr * ortho_update
