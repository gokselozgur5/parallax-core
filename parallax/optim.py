"""
Optimization algorithms for neural network training.

Implements AdamW (Loshchilov & Hutter, 2017) with decoupled weight decay,
first and second moment estimation, and bias correction.
"""

from typing import List, Tuple
import numpy as np
from parallax.nn import Parameter


class AdamW:
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        parameters: List[Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

        # Maintain first (m) and second (v) moment buffers
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
            if p.grad is None:
                continue

            grad = p.grad

            # 1. Decoupled weight decay update
            if self.weight_decay > 0.0 and p.data.ndim >= 2:
                # Apply weight decay only to 2D+ weights (matrices), omit 1D biases/norms
                p.data -= lr_t * self.weight_decay * p.data

            # 2. Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad

            # 3. Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad**2)

            # 4. Compute bias-corrected moments
            m_hat = self.m[i] / bias_corr1
            v_hat = self.v[i] / bias_corr2

            # 5. Parameter update
            step_update = lr_t * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= step_update
