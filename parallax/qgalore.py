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
import warnings
import numpy as np


class QGaLoreProjector:
    """Computes low-rank subspace basis and handles INT8 quantized gradient projection."""

    def __init__(self, rank: int, update_interval: int = 100):
        self.rank = rank
        self.effective_rank = rank
        self.update_interval = update_interval
        self.step_count = 0
        self.U: Optional[np.ndarray] = None
        self.V_T: Optional[np.ndarray] = None

    def update_basis(self, grad: np.ndarray):
        """Recompute low-rank SVD projection basis U from gradient.

        If rank exceeds min(M, N), effective_rank is clamped to min(M, N).
        """
        max_rank = min(grad.shape[0], grad.shape[1])
        if self.rank > max_rank:
            warnings.warn(
                f"rank ({self.rank}) exceeds min(M, N) ({max_rank}); clamping effective_rank to {max_rank}",
                UserWarning,
                stacklevel=2,
            )
        effective_rank = min(self.rank, max_rank)
        self.effective_rank = effective_rank
        U, _, Vt = np.linalg.svd(grad, full_matrices=False)
        self.U = U[:, :effective_rank].astype(np.float32)
        self.V_T = Vt[:effective_rank, :].astype(np.float32)

    def project_and_quantize(
        self, grad: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Project gradient into low-rank subspace and apply INT8 quantization.

        When M >= N, applies right-sided projection: P = grad @ V_T.T (shape M x rank).
        When M < N, applies left-sided projection: P = U.T @ grad (shape rank x N).
        Returns quantized integer array and scaling factor.
        """
        if self.U is None or self.V_T is None or (self.step_count % self.update_interval == 0):
            self.update_basis(grad)
        self.step_count += 1

        M, N = grad.shape[0], grad.shape[1]
        if M >= N:
            assert self.V_T is not None
            proj = np.dot(grad, self.V_T.T)
        else:
            assert self.U is not None
            proj = np.dot(self.U.T, grad)

        # Per-tensor symmetric INT8 quantization
        abs_max = float(np.max(np.abs(proj))) + 1e-8
        scale = abs_max / 127.0
        quantized = np.clip(np.round(proj / scale), -128, 127).astype(np.int8)

        return quantized, scale

    def deproject(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Dequantize subspace and reconstruct low-rank gradient approximation."""
        dequant = quantized.astype(np.float32) * scale
        if dequant.shape[1] == self.effective_rank and self.V_T is not None and dequant.shape[0] != self.effective_rank:
            return np.dot(dequant, self.V_T)
        if self.U is not None and dequant.shape[0] == self.effective_rank and (self.V_T is None or dequant.shape[1] != self.effective_rank):
            return np.dot(self.U, dequant)
        # In square case where M == N and M == effective_rank or both match:
        if self.V_T is not None and dequant.shape[1] == self.V_T.shape[0]:
            return np.dot(dequant, self.V_T)
        assert self.U is not None
        return np.dot(self.U, dequant)


from typing import List, Dict
from parallax.nn import Parameter


class QGaLoreAdamW:
    """Trainer-compatible AdamW optimizer using Q-GaLore projection for 2D weight matrices.

    Maintains INT8 low-rank project memory savings while plugging directly into ParallaxLM / Trainer.
    """

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rank: int = 16,
        update_interval: int = 100,
    ):
        self.params = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.rank = rank
        self.update_interval = update_interval
        self.step_count = 0

        self.projectors: Dict[int, QGaLoreProjector] = {}
        for p in self.params:
            if p.data.ndim == 2:
                self.projectors[id(p)] = QGaLoreProjector(rank=self.rank, update_interval=self.update_interval)

        self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.step_count += 1
        lr_t = self.lr
        bias_corr1 = 1.0 - (self.beta1**self.step_count)
        bias_corr2 = 1.0 - (self.beta2**self.step_count)

        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            grad = p.grad
            pid = id(p)

            # Apply Q-GaLore projection for 2D weight matrices
            if pid in self.projectors:
                proj = self.projectors[pid]
                q_sub, scale = proj.project_and_quantize(grad)
                effective_grad = proj.deproject(q_sub, scale)
            else:
                effective_grad = grad

            if self.weight_decay > 0.0 and p.data.ndim >= 2:
                p.data -= lr_t * self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * effective_grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (effective_grad**2)

            m_hat = self.m[i] / bias_corr1
            v_hat = self.v[i] / bias_corr2

            step_update = lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= step_update
