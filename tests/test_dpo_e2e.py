"""
End-to-end tests for Direct Preference Optimization (DPO) alignment loop.
"""

import numpy as np

from parallax.generator import CharTokenizer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.rl import compute_sequence_log_probs, dpo_loss
from scripts.train_dpo_preference import (
    build_synthetic_preference_dataset,
    prepare_dpo_batch,
    run_dpo_training,
)


def test_dpo_loss_decreases_and_implicit_margin_increases():
    """Verify that a 10-step DPO training run decreases loss and monotonically increases implicit margin."""
    history = run_dpo_training(steps=10, beta=0.1, lr=0.01, seed=42)

    losses = history["loss"]
    margins = history["margin"]
    r_chosen = history["r_chosen"]
    r_rejected = history["r_rejected"]

    assert len(losses) == 10
    # Initial loss is exactly log(2) ~= 0.6931 because policy and reference start identical
    assert np.isclose(losses[0], np.log(2.0), atol=1e-3)
    assert np.isclose(margins[0], 0.0, atol=1e-3)

    # Loss must decrease substantially
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.05

    # Final margin must be positive and greater than initial
    assert margins[-1] > margins[0]
    assert margins[-1] > 5.0

    # Chosen implicit reward must exceed rejected implicit reward
    assert r_chosen[-1] > r_rejected[-1]


def test_dpo_chosen_log_probs_increase_relative_to_rejected():
    """Verify that policy updates increase chosen token log-probabilities relative to rejected completions."""
    np.random.seed(1337)
    pairs = build_synthetic_preference_dataset()
    corpus = " ".join([p["prompt"] + " " + p["chosen"] + " " + p["rejected"] for p in pairs])
    tokenizer = CharTokenizer(corpus)

    config = ParallaxLMConfig(
        vocab_size=max(tokenizer.vocab_size, 64),
        embed_dim=32,
        num_heads=2,
        num_kv_heads=2,
        num_layers=2,
        max_seq_len=128,
        use_rope=True,
    )

    policy = ParallaxLM(config)
    ref = ParallaxLM(config)

    for p_ref, p_pol in zip(ref.parameters(), policy.parameters()):
        p_ref.data = p_pol.data.copy()

    in_c, tgt_c, mask_c, in_r, tgt_r, mask_r = prepare_dpo_batch(pairs, tokenizer)
    ref_c_logits = ref(in_c).data.copy()
    ref_r_logits = ref(in_r).data.copy()

    # Initial log-probabilities before DPO training
    init_pc = policy(in_c)
    init_pr = policy(in_r)
    init_lp_w, _ = compute_sequence_log_probs(init_pc, tgt_c, mask_c)
    init_lp_l, _ = compute_sequence_log_probs(init_pr, tgt_r, mask_r)
    init_log_ratio = init_lp_w - init_lp_l

    optimizer = AdamW(policy.parameters(), lr=0.01, weight_decay=0.0)

    for step in range(10):
        optimizer.zero_grad()
        p_c = policy(in_c)
        p_r = policy(in_r)

        loss, _, _ = dpo_loss(
            policy_chosen_logits=p_c,
            policy_rejected_logits=p_r,
            ref_chosen_logits=ref_c_logits,
            ref_rejected_logits=ref_r_logits,
            chosen_targets=tgt_c,
            rejected_targets=tgt_r,
            beta=0.1,
            chosen_mask=mask_c,
            rejected_mask=mask_r,
        )
        optimizer.step()

    # Log-probabilities after 10 DPO steps
    final_pc = policy(in_c)
    final_pr = policy(in_r)
    final_lp_w, _ = compute_sequence_log_probs(final_pc, tgt_c, mask_c)
    final_lp_l, _ = compute_sequence_log_probs(final_pr, tgt_r, mask_r)
    final_log_ratio = final_lp_w - final_lp_l

    # For each sample in the synthetic dataset:
    # 1. The difference (log P(chosen) - log P(rejected)) must strictly increase
    assert np.all(final_log_ratio > init_log_ratio)

    # 2. Chosen sequence log-prob change must be higher than rejected sequence log-prob change
    delta_w = final_lp_w - init_lp_w
    delta_l = final_lp_l - init_lp_l
    assert np.all(delta_w > delta_l)
