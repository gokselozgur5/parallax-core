"""Q-GaLore: Quantized Gradient Low-Rank Projection Optimizer.

Reference: Zhang et al., 2024 - Q-GaLore: Quantized GaLore with Low-Rank Projection.
Projects full-rank weight gradients into low-rank SVD subspaces and quantizes
the projected states to 8-bit integers to drastically reduce optimizer memory.

Integration contract:
    Standalone mathematical gradient projector and quantizer. Operates on raw
    NumPy gradient arrays. Not integrated into parallax.optim.AdamW or the
    computational autograd graph.
"""

from typing import Tuple, Optional
import numpy as np


class QGaLoreProjector:
    """Computes low-rank subspace basis and handles INT8 quantized gradient projection."""

    def __init__(self, rank: int, update_interval: int = 100):
        self.rank = rank
        self.update_interval = update_interval
        self.step_count = 0
        self.U: Optional[np.ndarray] = None
        self.V_T: Optional[np.ndarray] = None


    def update_basis(self, grad: np.ndarray):
        """Recompute low-rank SVD projection basis U from gradient."""
        max_rank = min(grad.shape[0], grad.shape[1])
        if self.rank > max_rank:
            raise ValueError(f"rank ({self.rank}) cannot exceed min(M, N) ({max_rank})")
        effective_rank = self.rank
        U, _, Vt = np.linalg.svd(grad, full_matrices=False)
        self.U = U[:, :effective_rank].astype(np.float32)
        self.V_T = Vt[:effective_rank, :].astype(np.float32)

    def project_and_quantize(
        self, grad: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Project gradient into low-rank subspace and apply INT8 quantization.

        P = U.T @ grad
        Returns quantized integer array and scaling factor.
        """
        if self.U is None or (self.step_count % self.update_interval == 0):
            self.update_basis(grad)
        self.step_count += 1

        # Project to low-rank subspace: shape (rank, N)
        proj = np.dot(self.U.T, grad)

        # Per-tensor symmetric INT8 quantization
        abs_max = float(np.max(np.abs(proj))) + 1e-8
        scale = abs_max / 127.0
        quantized = np.clip(np.round(proj / scale), -128, 127).astype(np.int8)

        return quantized, scale

    def deproject(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize subspace and reconstruct low-rank gradient approximation."""
        dequant = quantized.astype(np.float32) * scale
        return np.dot(self.U, dequant)
