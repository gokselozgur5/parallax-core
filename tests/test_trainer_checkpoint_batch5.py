"""
Unit tests for Trainer and Checkpoint robustness fixes (Batch 5).

Covers:
- Issue #245: Scheduler step-0 LR applied before first optimizer step.
- Issue #253: load_checkpoint binds restored optimizer state to returned model parameters.
- Issue #254: save_checkpoint / load_checkpoint round-trip with and without explicit extension.
"""

import os
import tempfile
import numpy as np
import pytest

from parallax.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW, CosineAnnealingWithWarmup
from parallax.trainer import Trainer, TrainerConfig


def dummy_data_loader(batch_size: int, seq_len: int):
    """Infinite generator of dummy tokens for trainer step verification."""
    pattern = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    x_fixed = np.tile(pattern[:seq_len], (batch_size, 1))
    y_fixed = np.tile(pattern[1 : seq_len + 1], (batch_size, 1))
    while True:
        yield x_fixed.copy(), y_fixed.copy()


def test_scheduler_step0_lr_applied_before_first_optimizer_step():
    """Verify scheduler step-0 LR is applied before step 1 optimizer update (#245)."""
    config = ParallaxLMConfig(
        vocab_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
        tie_weights=True,
    )
    model = ParallaxLM(config)
    initial_base_lr = 0.1
    min_lr = 0.01
    warmup_steps = 5

    opt = AdamW(model.parameters(), lr=initial_base_lr, weight_decay=0.0)
    scheduler = CosineAnnealingWithWarmup(
        opt,
        warmup_steps=warmup_steps,
        total_steps=20,
        min_lr=min_lr,
        base_lr=initial_base_lr,
    )

    # 1. Scheduler __init__ must have set optimizer.lr to get_lr(0) immediately
    step0_lr = scheduler.get_lr(0)
    assert opt.lr == step0_lr
    assert opt.lr == min_lr

    train_cfg = TrainerConfig(
        max_steps=5,
        grad_accum_steps=1,
        max_grad_norm=None,
        eval_interval=10,
    )
    loader = dummy_data_loader(batch_size=2, seq_len=4)
    trainer = Trainer(
        model=model,
        optimizer=opt,
        config=train_cfg,
        train_loader=loader,
        scheduler=scheduler,
    )

    history = trainer.train()

    # Step 1 should record step-0 LR (the LR used during optimizer.step)
    assert len(history) == 5
    assert history[0]["step"] == 1.0
    assert history[0]["lr"] == step0_lr
    assert history[1]["step"] == 2.0
    assert history[1]["lr"] == scheduler.get_lr(1)
    assert history[2]["lr"] == scheduler.get_lr(2)


def test_checkpoint_restore_binds_optimizer_to_returned_model():
    """Verify load_checkpoint binds restored optimizer state to returned model (#253)."""
    config = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
        tie_weights=False,
    )
    model = ParallaxLM(config)
    opt = AdamW(model.parameters(), lr=0.02)

    # Perform a dummy training step to populate m and v
    x = np.array([[1, 2, 3]])
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    opt.step()

    assert opt.step_count == 1
    assert any(np.any(m != 0) for m in opt.m)
    assert any(np.any(v != 0) for v in opt.v)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "model_and_opt.npz")
        saved_path = save_checkpoint(ckpt_path, model, optimizer=opt, step=10)

        # Restore into a fresh optimizer instance
        restored_opt = AdamW([], lr=0.0)
        restored_model, restored_step = load_checkpoint(saved_path, optimizer=restored_opt)

        assert restored_step == 10
        assert restored_opt.step_count == 1
        assert restored_opt.lr == 0.02
        assert len(restored_opt.params) == len(restored_model.parameters())

        # Parameters must be exact identical object references to restored_model
        for opt_p, model_p in zip(restored_opt.params, restored_model.parameters()):
            assert opt_p is model_p

        # Buffer moments must match original saved moments
        for orig_m, rest_m in zip(opt.m, restored_opt.m):
            assert np.allclose(orig_m, rest_m)
        for orig_v, rest_v in zip(opt.v, restored_opt.v):
            assert np.allclose(orig_v, rest_v)

        # Subsequent optimizer step on restored model modifies restored model parameters
        param_before = [p.data.copy() for p in restored_model.parameters()]
        loss2 = restored_model(x).sum()
        loss2.backward()
        restored_opt.step()

        for p_before, p in zip(param_before, restored_model.parameters()):
            assert not np.allclose(p_before, p.data)


def test_save_and_load_checkpoint_extension_normalization():
    """Verify save_checkpoint and load_checkpoint handle paths with or without .npz (#254)."""
    config = ParallaxLMConfig(
        vocab_size=16,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        max_seq_len=8,
    )
    model = ParallaxLM(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case 1: path without extension
        raw_path = os.path.join(tmpdir, "checkpoint_no_ext")
        returned_path = save_checkpoint(raw_path, model, step=7)
        assert returned_path == f"{raw_path}.npz"
        assert os.path.exists(returned_path)

        # load_checkpoint using raw path without extension
        loaded_model1, step1 = load_checkpoint(raw_path)
        assert step1 == 7
        assert isinstance(loaded_model1, ParallaxLM)

        # load_checkpoint using explicit .npz extension
        loaded_model2, step2 = load_checkpoint(returned_path)
        assert step2 == 7

        # Case 2: path with explicit extension
        explicit_path = os.path.join(tmpdir, "checkpoint_explicit.npz")
        ret2 = save_checkpoint(explicit_path, model, step=15)
        assert ret2 == explicit_path
        assert os.path.exists(explicit_path)

        loaded_model3, step3 = load_checkpoint(explicit_path)
        assert step3 == 15
