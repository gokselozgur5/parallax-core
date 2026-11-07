"""
Modular Language Model Trainer with Gradient Accumulation.

Encapsulates the standard LLM pretraining / fine-tuning loop:
- Micro-batch gradient accumulation (simulates large effective batch size)
- Global gradient norm clipping
- Learning rate schedule stepping
- Evaluation loop and perplexity tracking
- Checkpoint saving callbacks
"""

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import numpy as np

from parallax.checkpoint import save_checkpoint
from parallax.loss import compute_perplexity, cross_entropy_loss
from parallax.models.gpt import ParallaxLM
from parallax.optim import AdamW, LRScheduler, clip_grad_norm_


@dataclass
class TrainerConfig:
    """Configuration for Parallax LM Trainer."""
    max_steps: int = 100
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    label_smoothing: float = 0.0
    eval_interval: int = 20
    eval_steps: int = 5
    checkpoint_interval: Optional[int] = None
    checkpoint_prefix: str = "checkpoint"


class Trainer:
    """First-principles Trainer for ParallaxLM."""

    def __init__(
        self,
        model: ParallaxLM,
        optimizer: AdamW,
        config: TrainerConfig,
        train_loader: Iterator[Tuple[np.ndarray, np.ndarray]],
        val_loader: Optional[Iterator[Tuple[np.ndarray, np.ndarray]]] = None,
        scheduler: Optional[LRScheduler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.history: List[Dict[str, float]] = []

    def evaluate(self) -> float:
        """Evaluates model over val_loader, returning mean cross-entropy loss."""
        if self.val_loader is None:
            return 0.0

        val_losses: List[float] = []
        for _ in range(self.config.eval_steps):
            x, y = next(self.train_loader if self.val_loader is None else self.val_loader)
            logits = self.model(x)
            loss = cross_entropy_loss(logits, y, label_smoothing=0.0)
            val_losses.append(loss.data)

        return float(np.mean(val_losses))

    def train(self) -> List[Dict[str, float]]:
        """Executes training loop with gradient accumulation."""
        cfg = self.config
        model = self.model
        opt = self.optimizer

        opt.zero_grad()
        accum_loss = 0.0

        for step in range(1, cfg.max_steps + 1):
            t0 = time.perf_counter()

            # Inner micro-batch accumulation loop
            micro_loss_sum = 0.0
            for _ in range(cfg.grad_accum_steps):
                x_batch, y_batch = next(self.train_loader)
                logits = model(x_batch)
                loss = cross_entropy_loss(
                    logits,
                    y_batch,
                    label_smoothing=cfg.label_smoothing,
                )
                # Scale loss by 1 / grad_accum_steps for accumulation
                scaled_loss = loss * (1.0 / cfg.grad_accum_steps)
                scaled_loss.backward()
                micro_loss_sum += loss.data

            step_loss = micro_loss_sum / cfg.grad_accum_steps

            # Gradient clipping
            grad_norm = 0.0
            if cfg.max_grad_norm is not None:
                grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            # Optimizer & Scheduler step
            opt.step()
            opt.zero_grad()

            current_lr = opt.lr
            if self.scheduler is not None:
                current_lr = self.scheduler.step()

            step_time = time.perf_counter() - t0
            metrics: Dict[str, float] = {
                "step": float(step),
                "loss": step_loss,
                "ppl": compute_perplexity(step_loss),
                "grad_norm": grad_norm,
                "lr": current_lr,
                "dt_ms": step_time * 1000.0,
            }

            # Evaluation check
            if self.val_loader is not None and step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                metrics["val_loss"] = val_loss
                metrics["val_ppl"] = compute_perplexity(val_loss)

            # Checkpoint callback
            if cfg.checkpoint_interval is not None and step % cfg.checkpoint_interval == 0:
                save_checkpoint(f"{cfg.checkpoint_prefix}_step_{step}.npz", model, opt, step=step)

            self.history.append(metrics)

        return self.history
