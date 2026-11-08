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
