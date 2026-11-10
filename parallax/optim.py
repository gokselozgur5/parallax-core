"""
Optimization algorithms, gradient management, and learning rate schedules.

Implements:
- AdamW (Loshchilov & Hutter, 2017) with decoupled weight decay
- Global gradient norm clipping (`clip_grad_norm_`)
- Learning rate schedulers: CosineAnnealingWithWarmup, LinearWarmup
"""

import math
from typing import List, Optional, Tuple
import numpy as np
from parallax.nn import Parameter


def clip_grad_norm_(parameters: List[Parameter], max_norm: float, eps: float = 1e-6) -> float:
    """Clips the global norm of parameter gradients.

    Args:
        parameters: List of Parameters whose gradients should be clipped.
        max_norm: Upper bound on the global gradient L2 norm.
        eps: Small constant to prevent division by zero.

    Returns:
        Total L2 norm of the gradients prior to clipping.
    """
    total_sq_sum = 0.0
    grads = [p.grad for p in parameters if p.requires_grad and p.grad is not None]

    if not grads:
        return 0.0

    for g in grads:
        total_sq_sum += float(np.sum(g**2))

    total_norm = math.sqrt(total_sq_sum)
    clip_coef = max_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for g in grads:
            g *= clip_coef

    return total_norm


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: "AdamW", base_lr: Optional[float] = None):
        self.optimizer = optimizer
        self.base_lr = base_lr if base_lr is not None else optimizer.lr
        self.step_count = 0
        try:
            self.optimizer.lr = self.get_lr(0)
        except (NotImplementedError, AttributeError):
            pass

    def get_lr(self, step: int) -> float:
        raise NotImplementedError

    def step(self) -> float:
        self.step_count += 1
        lr = self.get_lr(self.step_count)
        self.optimizer.lr = lr
        return lr


class CosineAnnealingWithWarmup(LRScheduler):
    """Cosine learning rate decay with linear warmup.

    - Warmup phase: linearly scales LR from min_lr (or 0) to base_lr
    - Cosine decay phase: cosine decay from base_lr down to min_lr
    - After total_steps: stays at min_lr
    """

    def __init__(
        self,
        optimizer: "AdamW",
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        assert total_steps >= warmup_steps, "total_steps must be >= warmup_steps"
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, base_lr=base_lr)
        self.optimizer.lr = self.get_lr(0)

    def get_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            # Linear warmup
            if self.warmup_steps == 0:
                return self.base_lr
            return self.min_lr + (self.base_lr - self.min_lr) * (step / self.warmup_steps)

        if step > self.total_steps:
            return self.min_lr

        # Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


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
            if not p.requires_grad or p.grad is None:
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
