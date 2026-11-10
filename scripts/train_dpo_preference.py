"""
Train ParallaxLM using Direct Preference Optimization (DPO) on synthetic preference pairs.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np

# Ensure parallax is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parallax.generator import CharTokenizer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.rl import dpo_loss


def build_synthetic_preference_dataset() -> List[Dict[str, str]]:
    """Build synthetic prompt, chosen, and rejected preference pairs."""
    return [
        {
            "prompt": "Parallax",
            "chosen": "Parallax creates clarity and deep insights.",
            "rejected": "Parallax causes confusion and broken states.",
        },
        {
            "prompt": "Neural network",
            "chosen": "Neural network optimizes objective cleanly.",
            "rejected": "Neural network explodes gradients completely.",
        },
        {
            "prompt": "Gradient descent",
            "chosen": "Gradient descent converges stably to minimum.",
            "rejected": "Gradient descent diverges wildly into NaN.",
        },
        {
            "prompt": "Autonomous agent",
            "chosen": "Autonomous agent verifies evidence rigorously.",
            "rejected": "Autonomous agent hallucinates without check.",
        },
    ]


def prepare_dpo_batch(
    pairs: List[Dict[str, str]],
    tokenizer: CharTokenizer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize and pad preference pairs into batch tensors with prompt masking."""
    encoded_pairs = []
    max_len = 0
    for p in pairs:
        prompt_ids = tokenizer.encode(p["prompt"])
        c_ids = tokenizer.encode(p["chosen"])
        r_ids = tokenizer.encode(p["rejected"])
        if len(c_ids) < 2:
            c_ids = c_ids + [0]
        if len(r_ids) < 2:
            r_ids = r_ids + [0]
        max_len = max(max_len, len(c_ids), len(r_ids))
        encoded_pairs.append((prompt_ids, c_ids, r_ids))

    batch_size = len(pairs)
    c_padded = np.zeros((batch_size, max_len), dtype=np.int64)
    r_padded = np.zeros((batch_size, max_len), dtype=np.int64)
    mask_c = np.zeros((batch_size, max_len - 1), dtype=np.float64)
    mask_r = np.zeros((batch_size, max_len - 1), dtype=np.float64)

    for i, (p_ids, c_ids, r_ids) in enumerate(encoded_pairs):
        c_padded[i, : len(c_ids)] = c_ids
        r_padded[i, : len(r_ids)] = r_ids

        p_len = max(0, len(p_ids) - 1)
        mask_c[i, p_len : len(c_ids) - 1] = 1.0
        if np.sum(mask_c[i]) == 0.0:
            mask_c[i, : len(c_ids) - 1] = 1.0

        mask_r[i, p_len : len(r_ids) - 1] = 1.0
        if np.sum(mask_r[i]) == 0.0:
            mask_r[i, : len(r_ids) - 1] = 1.0

    in_c = c_padded[:, :-1]
    tgt_c = c_padded[:, 1:]
    in_r = r_padded[:, :-1]
    tgt_r = r_padded[:, 1:]

    return in_c, tgt_c, mask_c, in_r, tgt_r, mask_r


def run_dpo_training(
    steps: int = 10,
    beta: float = 0.1,
    lr: float = 0.01,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Runs a 10-step DPO optimization loop and tracks loss and implicit rewards."""
    np.random.seed(seed)
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

    policy_model = ParallaxLM(config)
    ref_model = ParallaxLM(config)

    # Initialize reference model with exact identical initial weights
    for p_ref, p_pol in zip(ref_model.parameters(), policy_model.parameters()):
        p_ref.data = p_pol.data.copy()

    in_c, tgt_c, mask_c, in_r, tgt_r, mask_r = prepare_dpo_batch(pairs, tokenizer)
    ref_c_logits = ref_model(in_c).data.copy()
    ref_r_logits = ref_model(in_r).data.copy()

    optimizer = AdamW(policy_model.parameters(), lr=lr, weight_decay=0.0)

    history: Dict[str, List[float]] = {
        "step": [],
        "loss": [],
        "r_chosen": [],
        "r_rejected": [],
        "margin": [],
    }

    print(f"Starting DPO Training: {steps} steps, beta={beta}, lr={lr}, batch_size={len(pairs)}")
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        p_c_logits = policy_model(in_c)
        p_r_logits = policy_model(in_r)

        loss, r_w, r_l = dpo_loss(
            policy_chosen_logits=p_c_logits,
            policy_rejected_logits=p_r_logits,
            ref_chosen_logits=ref_c_logits,
            ref_rejected_logits=ref_r_logits,
            chosen_targets=tgt_c,
            rejected_targets=tgt_r,
            beta=beta,
            chosen_mask=mask_c,
            rejected_mask=mask_r,
        )

        optimizer.step()
        margin = r_w - r_l

        history["step"].append(step)
        history["loss"].append(float(loss))
        history["r_chosen"].append(float(r_w))
        history["r_rejected"].append(float(r_l))
        history["margin"].append(float(margin))

        print(
            f"Step {step:02d} | Loss: {loss:.4f} | "
            f"r_chosen: {r_w:+.4f} | r_rejected: {r_l:+.4f} | Margin: {margin:+.4f}"
        )

    return history


if __name__ == "__main__":
    run_dpo_training(steps=10)
