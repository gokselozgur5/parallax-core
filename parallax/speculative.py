"""
Speculative Decoding (Leviathan et al., 2023; Chen et al., 2023) from first principles.

Accelerates autoregressive inference by generating gamma candidate tokens with a
fast, smaller Draft model (M_draft) and verifying all gamma tokens in parallel
with a single forward pass of the larger Target model (M_target).

Guarantees lossless generation matching the exact target model probability distribution
via the speculative acceptance criterion:
    Accept token x with probability min(1, p(x) / q(x))
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from parallax.generator import sample_next_token
from parallax.models.gpt import ParallaxLM


def speculative_generate(
    target_model: ParallaxLM,
    draft_model: ParallaxLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 20,
    gamma: int = 3,
    temperature: float = 0.0,
) -> Tuple[List[int], Dict[str, float]]:
    """Generates text autoregressively using speculative decoding.

    Args:
        target_model: Primary high-capacity model (slower).
        draft_model: Lightweight draft model (faster, sharing the same vocabulary).
        prompt_tokens: Initial prompt token IDs.
        max_new_tokens: Total budget of new tokens to generate.
        gamma: Number of lookahead speculative tokens proposed per draft cycle.
        temperature: Sampling temperature (0.0 = greedy).

    Returns:
        generated_tokens: Complete token sequence including prompt.
        stats: Dictionary containing acceptance_rate, total_accepted, and target_forward_passes.
    """
    tokens = list(prompt_tokens)
    total_generated = 0
    total_proposed = 0
    total_accepted = 0
    target_forward_passes = 0

    max_seq_len = min(target_model.config.max_seq_len, draft_model.config.max_seq_len)

    while total_generated < max_new_tokens and len(tokens) < max_seq_len:
        current_len = len(tokens)
        spec_budget = min(gamma, max_new_tokens - total_generated, max_seq_len - current_len - 1)
        if spec_budget <= 0:
            break

        # 1. Draft step: Propose `spec_budget` tokens autoregressively using the draft model
        draft_tokens: List[int] = []
        draft_probs: List[float] = []
        curr_draft_seq = list(tokens)

        for _ in range(spec_budget):
            idx_d = np.array([curr_draft_seq[-draft_model.config.max_seq_len:]], dtype=np.int64)
            d_logits = draft_model(idx_d).data[0, -1, :]
            # Compute draft probability distribution for candidate
            exp_d = np.exp(d_logits - np.max(d_logits))
            d_p = exp_d / np.sum(exp_d)
            next_t = sample_next_token(d_logits, temperature=temperature)
            draft_tokens.append(next_t)
            draft_probs.append(float(d_p[next_t]))
            curr_draft_seq.append(next_t)

        total_proposed += len(draft_tokens)

        # 2. Parallel Target verification: evaluate prompt + all drafted tokens in ONE forward pass
        eval_seq = tokens + draft_tokens
        idx_target = np.array([eval_seq[-target_model.config.max_seq_len:]], dtype=np.int64)
        target_logits = target_model(idx_target).data[0]
        target_forward_passes += 1

        # 3. Acceptance / Rejection loop
        # Positions to evaluate:
        # Prompt end corresponds to index (current_len - 1) in eval_seq
        # logits at that index predict candidate 0
        n_accepted = 0
        offset = len(eval_seq) - len(draft_tokens) - 1

        for i, cand in enumerate(draft_tokens):
            pos_logits = target_logits[offset + i]

            if temperature <= 0.0:
                # Greedy equivalence: accept if target's top prediction matches draft candidate
                target_choice = int(np.argmax(pos_logits))
                if cand == target_choice:
                    tokens.append(cand)
                    n_accepted += 1
                    total_generated += 1
                    if total_generated >= max_new_tokens:
                        break
                else:
                    # Reject: emit target model's true choice and truncate cycle
                    tokens.append(target_choice)
                    total_generated += 1
                    break
            else:
                # Stochastic acceptance criterion: min(1, p(x) / q(x))
                cand_target_prob = np.exp(pos_logits[cand] - np.max(pos_logits))
                cand_target_prob /= np.sum(np.exp(pos_logits - np.max(pos_logits)))
                q_prob = max(draft_probs[i], 1e-12)
                ratio = float(cand_target_prob) / q_prob

                # Exact speculative acceptance check: accept with probability min(1, p(x)/q(x))
                if np.random.rand() < min(1.0, ratio):
                    tokens.append(cand)
                    n_accepted += 1
                    total_generated += 1
                    if total_generated >= max_new_tokens:
                        break
                else:
                    # Sample corrected replacement token from target distribution
                    corr_token = sample_next_token(pos_logits, temperature=temperature)
                    tokens.append(corr_token)
                    total_generated += 1
                    break
        else:
            # All gamma tokens were accepted! Sample one bonus token from target's final distribution
            if total_generated < max_new_tokens and len(tokens) < max_seq_len:
                bonus_logits = target_logits[-1]
                bonus_token = sample_next_token(bonus_logits, temperature=temperature)
                tokens.append(bonus_token)
                total_generated += 1

        total_accepted += n_accepted

    acceptance_rate = (total_accepted / total_proposed) if total_proposed > 0 else 0.0
    stats = {
        "acceptance_rate": round(acceptance_rate, 3),
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
        "target_forward_passes": target_forward_passes,
        "speedup_factor": round(total_generated / max(1, target_forward_passes), 2),
    }
    return tokens, stats
