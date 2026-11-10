"""
Speculative Decoding (Leviathan et al., 2023; Chen et al., 2023) from first principles.

Accelerates autoregressive inference by generating gamma candidate tokens with a
fast, smaller Draft model (M_draft) and verifying all gamma tokens in parallel
with a single forward pass of the larger Target model (M_target).

Guarantees lossless generation matching the exact target model probability distribution
via the speculative acceptance criterion:
    Accept token x with probability min(1, p(x) / q(x))
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from parallax.generator import sample_next_token
from parallax.models.gpt import ParallaxLM
from parallax.tensor import no_grad


def _compute_token_probs(
    logits: np.ndarray,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> np.ndarray:
    """Computes filtered probability distribution over vocabulary."""
    if temperature is not None:
        if np.isnan(temperature) or np.isinf(temperature) or temperature < 0:
            raise ValueError(f"Invalid temperature: {temperature}. Must be finite and non-negative.")
        if temperature < 1e-7 or temperature == 0:
            probs = np.zeros_like(logits, dtype=np.float64)
            probs[int(np.argmax(logits))] = 1.0
            return probs
    else:
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[int(np.argmax(logits))] = 1.0
        return probs

    scaled = logits.copy().astype(np.float64) / temperature
    if top_k is not None and 0 < top_k < len(scaled):
        indices_to_remove = np.argsort(scaled)[:-top_k]
        scaled[indices_to_remove] = -np.inf

    exp_l = np.exp(scaled - np.max(scaled))
    probs = exp_l / np.sum(exp_l)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        sorted_indices_to_remove = cumulative > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        s = np.sum(probs)
        if s > 0:
            probs = probs / s
        else:
            probs[sorted_indices[0]] = 1.0
    return probs


def speculative_generate(
    target_model: ParallaxLM,
    draft_model: ParallaxLM,
    prompt_tokens: List[int],
    max_new_tokens: int = 20,
    gamma: int = 3,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """Generates text autoregressively using speculative decoding.

    Args:
        target_model: Primary high-capacity model (slower).
        draft_model: Lightweight draft model (faster, sharing the same vocabulary).
        prompt_tokens: Initial prompt token IDs.
        max_new_tokens: Total budget of new tokens to generate.
        gamma: Number of lookahead speculative tokens proposed per draft cycle.
        temperature: Sampling temperature (0.0 = greedy).
        top_k: Optional top-k filtering limit.
        top_p: Optional top-p nucleus filtering threshold.
        eos_token_id: Optional end-of-sequence token ID.

    Returns:
        generated_tokens: Complete token sequence including prompt.
        stats: Dictionary containing acceptance_rate, total_accepted, target_forward_passes,
               effective speedup factor, accepted_tokens_per_step, and wall-clock elapsed_time_sec.
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")

    tokens = list(prompt_tokens)
    total_generated = 0
    total_proposed = 0
    total_accepted = 0
    target_forward_passes = 0
    start_time = time.perf_counter()

    max_seq_len = min(target_model.config.max_seq_len, draft_model.config.max_seq_len)

    while total_generated < max_new_tokens and len(tokens) < max_seq_len:
        current_len = len(tokens)
        spec_budget = min(gamma, max_new_tokens - total_generated, max_seq_len - current_len)
        if spec_budget <= 0:
            break

        # 1. Draft step: Propose `spec_budget` tokens autoregressively using the draft model
        draft_tokens: List[int] = []
        draft_prob_dists: List[np.ndarray] = []
        curr_draft_seq = list(tokens)
        hit_eos_in_draft = False

        for _ in range(spec_budget):
            idx_d = np.array([curr_draft_seq[-draft_model.config.max_seq_len:]], dtype=np.int64)
            with no_grad():
                d_logits = draft_model(idx_d).data[0, -1, :]
            # Compute draft probability distribution for candidate
            d_p = _compute_token_probs(d_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            next_t = sample_next_token(d_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_tokens.append(next_t)
            draft_prob_dists.append(d_p)
            curr_draft_seq.append(next_t)
            if eos_token_id is not None and next_t == eos_token_id:
                hit_eos_in_draft = True
                break

        total_proposed += len(draft_tokens)

        # 2. Parallel Target verification: evaluate prompt + all drafted tokens in ONE forward pass
        eval_seq = tokens + draft_tokens
        idx_target = np.array([eval_seq[-target_model.config.max_seq_len:]], dtype=np.int64)
        with no_grad():
            target_logits = target_model(idx_target).data[0]
        target_forward_passes += 1

        # 3. Acceptance / Rejection loop
        # Positions to evaluate:
        # Prompt end corresponds to index (current_len - 1) in eval_seq
        # logits at that index predict candidate 0
        n_accepted = 0
        offset = len(eval_seq) - len(draft_tokens) - 1
        stopped_early = False
        all_accepted = True

        for i, cand in enumerate(draft_tokens):
            pos_logits = target_logits[offset + i]

            if temperature <= 0.0:
                # Greedy equivalence: accept if target's top prediction matches draft candidate
                target_choice = int(np.argmax(pos_logits))
                if cand == target_choice:
                    tokens.append(cand)
                    n_accepted += 1
                    total_generated += 1
                    if (eos_token_id is not None and cand == eos_token_id) or total_generated >= max_new_tokens:
                        stopped_early = True
                        all_accepted = False
                        break
                else:
                    # Reject: emit target model's true choice and truncate cycle
                    tokens.append(target_choice)
                    total_generated += 1
                    if eos_token_id is not None and target_choice == eos_token_id:
                        stopped_early = True
                    all_accepted = False
                    break
            else:
                # Stochastic acceptance criterion: min(1, p(x) / q(x))
                t_p = _compute_token_probs(pos_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                cand_target_prob = float(t_p[cand])
                q_prob = max(float(draft_prob_dists[i][cand]), 1e-12)
                ratio = cand_target_prob / q_prob

                # Exact speculative acceptance check: accept with probability min(1, p(x)/q(x))
                if np.random.rand() < min(1.0, ratio):
                    tokens.append(cand)
                    n_accepted += 1
                    total_generated += 1
                    if (eos_token_id is not None and cand == eos_token_id) or total_generated >= max_new_tokens:
                        stopped_early = True
                        all_accepted = False
                        break
                else:
                    # Sample corrected replacement token from normalized residual distribution
                    residual = np.maximum(0.0, t_p - draft_prob_dists[i])
                    res_sum = float(np.sum(residual))
                    if res_sum > 1e-12:
                        res_dist = residual / res_sum
                    else:
                        res_dist = t_p
                    corr_token = int(np.random.choice(len(res_dist), p=res_dist))
                    tokens.append(corr_token)
                    total_generated += 1
                    if eos_token_id is not None and corr_token == eos_token_id:
                        stopped_early = True
                    all_accepted = False
                    break
        else:
            # All gamma tokens were accepted! Sample one bonus token from target's final distribution
            if not hit_eos_in_draft and not stopped_early:
                if total_generated < max_new_tokens and len(tokens) < max_seq_len:
                    bonus_logits = target_logits[-1]
                    bonus_token = sample_next_token(bonus_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                    tokens.append(bonus_token)
                    total_generated += 1
                    if eos_token_id is not None and bonus_token == eos_token_id:
                        stopped_early = True

        total_accepted += n_accepted
        if stopped_early:
            break

    elapsed_time = time.perf_counter() - start_time
    acceptance_rate = (total_accepted / total_proposed) if total_proposed > 0 else 0.0
    accepted_per_step = round(total_accepted / max(1, target_forward_passes), 2)
    stats = {
        "acceptance_rate": round(acceptance_rate, 3),
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
        "target_forward_passes": target_forward_passes,
        "speedup_factor": round(total_generated / max(1, target_forward_passes), 2),
        "accepted_tokens_per_step": accepted_per_step,
        "elapsed_time_sec": round(elapsed_time, 5),
    }
    return tokens, stats
