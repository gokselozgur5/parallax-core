"""
Unit tests for Trainer with gradient accumulation and evaluation (M14).
"""

import itertools
import numpy as np
import pytest

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.lora import apply_lora_to_model
from parallax.optim import AdamW, CosineAnnealingWithWarmup
from parallax.trainer import Trainer, TrainerConfig


def dummy_data_loader(batch_size: int, seq_len: int):
    """Infinite generator of memorizable fixed sequence for overfitting verification."""
    # Fixed pattern [0, 1, 2, 3] repeated
    pattern = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    x_fixed = np.tile(pattern[:seq_len], (batch_size, 1))
    y_fixed = np.tile(pattern[1 : seq_len + 1], (batch_size, 1))
    while True:
        yield x_fixed.copy(), y_fixed.copy()


def test_trainer_gradient_accumulation_and_loss_decrease():
    np.random.seed(42)
    config = ParallaxLMConfig(
        vocab_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
        tie_weights=True,
    )
    model = ParallaxLM(config)
    opt = AdamW(model.parameters(), lr=0.08, weight_decay=0.0)

    train_cfg = TrainerConfig(
        max_steps=20,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        eval_interval=5,
        eval_steps=2,
    )

    train_gen = dummy_data_loader(batch_size=2, seq_len=4)
    val_gen = dummy_data_loader(batch_size=2, seq_len=4)

    scheduler = CosineAnnealingWithWarmup(opt, warmup_steps=2, total_steps=20, min_lr=0.01)

    trainer = Trainer(
        model=model,
        optimizer=opt,
        config=train_cfg,
        train_loader=train_gen,
        val_loader=val_gen,
        scheduler=scheduler,
    )

    history = trainer.train()

    assert len(history) == 20
    assert "loss" in history[0]
    assert "ppl" in history[0]
    assert "grad_norm" in history[0]
    assert "val_loss" in history[4]  # Step 5 has eval

    # Loss must trend downward significantly on fixed pattern
    initial_loss = history[0]["loss"]
    final_loss = history[-1]["loss"]
    assert final_loss < initial_loss * 0.5


def test_trainer_sets_model_training_mode_with_lora():
    config = ParallaxLMConfig(
        vocab_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
    )
    model = ParallaxLM(config)
    lora_layers = apply_lora_to_model(model, target_modules=("q_proj",), r=2, lora_dropout=0.5)
    model.eval()
    assert model.training is False
    assert all(not l.training for l in lora_layers)

    opt = AdamW(model.parameters(), lr=0.01)
    train_cfg = TrainerConfig(max_steps=2, grad_accum_steps=1, eval_interval=1, eval_steps=1)
    train_gen = dummy_data_loader(batch_size=2, seq_len=4)
    val_gen = dummy_data_loader(batch_size=2, seq_len=4)
    trainer = Trainer(model=model, optimizer=opt, config=train_cfg, train_loader=train_gen, val_loader=val_gen)

    trainer.train()
    assert model.training is True
    assert all(l.training for l in lora_layers)

    trainer.evaluate()
    assert model.training is False
    assert all(not l.training for l in lora_layers)

