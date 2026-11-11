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


def _eval_target_step(
    target_model: ParallaxLM,
    seq: List[int],
) -> np.ndarray:
    """Evaluates target model on seq respecting context limit."""
    idx = np.array([seq[-target_model.config.max_seq_len:]], dtype=np.int64)
    with no_grad():
        return target_model(idx).data[0]


def _verify_loop(
    target_model: ParallaxLM,
    tokens: List[int],
    draft_tokens: List[int],
    draft_prob_dists: List[np.ndarray],
    hit_eos_in_draft: bool,
    max_new_tokens: int,
    total_generated: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    eos_token_id: Optional[int],
) -> Tuple[int, int, int, bool]:
    """Verifies draft tokens sequentially against sliding window target."""
    n_acc = 0
    fwd = 0
    all_acc = True
    stopped = False
    for i, cand in enumerate(draft_tokens):
        pos_logits = _eval_target_step(target_model, tokens)[-1]
        fwd += 1
        is_acc, chosen = _verify_candidate(
            pos_logits, cand, draft_prob_dists[i], temperature, top_k, top_p
        )
        tokens.append(chosen)
        total_generated += 1
        if is_acc:
            n_acc += 1
        else:
            all_acc = False
        if (eos_token_id is not None and chosen == eos_token_id) or total_generated >= max_new_tokens:
            stopped = True
            break
        if not is_acc:
            break
    else:
        if not hit_eos_in_draft and all_acc and not stopped:
            if total_generated < max_new_tokens:
                bonus_logits = _eval_target_step(target_model, tokens)[-1]
                fwd += 1
                b_tok = sample_next_token(bonus_logits, temperature=temperature, top_k=top_k, top_p=top_p)
                tokens.append(b_tok)
                total_generated += 1
                if eos_token_id is not None and b_tok == eos_token_id:
                    stopped = True
    return n_acc, fwd, total_generated, stopped


def _verify_candidate(
    pos_logits: np.ndarray,
    cand: int,
    draft_prob: Optional[np.ndarray],
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[bool, int]:
    """Verifies a single draft candidate token against target logits."""
    if temperature <= 0.0:
        target_choice = int(np.argmax(pos_logits))
        return (cand == target_choice), (cand if cand == target_choice else target_choice)
    t_p = _compute_token_probs(pos_logits, temperature=temperature, top_k=top_k, top_p=top_p)
    q_p = draft_prob[cand] if draft_prob is not None else 1e-12
    ratio = float(t_p[cand]) / max(float(q_p), 1e-12)
    if np.random.rand() < min(1.0, ratio):
        return True, cand
    residual = np.maximum(0.0, t_p - (draft_prob if draft_prob is not None else 0.0))
    res_sum = float(np.sum(residual))
    res_dist = (residual / res_sum) if res_sum > 1e-12 else t_p
    return False, int(np.random.choice(len(res_dist), p=res_dist))


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
               target_pass_ratio (tokens per target pass), accepted_tokens_per_step,
               and wall-clock elapsed_time_sec.
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

    while total_generated < max_new_tokens:
        spec_budget = min(gamma, max_new_tokens - total_generated)
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

        # 2. Target verification:
        if len(tokens) + len(draft_tokens) > target_model.config.max_seq_len:
            n_acc, fwd, total_generated, stopped_early = _verify_loop(
                target_model,
                tokens,
                draft_tokens,
                draft_prob_dists,
                hit_eos_in_draft,
                max_new_tokens,
                total_generated,
                temperature,
                top_k,
                top_p,
                eos_token_id,
            )
            total_accepted += n_acc
            target_forward_passes += fwd
            if stopped_early:
                break
            continue

        # Parallel Target verification: evaluate prompt + all drafted tokens in ONE forward pass
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
            is_acc, chosen = _verify_candidate(
                pos_logits, cand, draft_prob_dists[i], temperature, top_k, top_p
            )
            tokens.append(chosen)
            total_generated += 1
            if is_acc:
                n_accepted += 1
            else:
                all_accepted = False
            if (eos_token_id is not None and chosen == eos_token_id) or total_generated >= max_new_tokens:
                stopped_early = True
                break
            if not is_acc:
                break
        else:
            # All gamma tokens were accepted! Sample one bonus token from target's final distribution
            if not hit_eos_in_draft and all_accepted and not stopped_early:
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
    target_pass_ratio = round(total_generated / max(1, target_forward_passes), 2)
    stats = {
        "acceptance_rate": round(acceptance_rate, 3),
        "total_accepted": total_accepted,
        "total_proposed": total_proposed,
        "target_forward_passes": target_forward_passes,
        "target_pass_ratio": target_pass_ratio,
        "accepted_tokens_per_step": accepted_per_step,
        "elapsed_time_sec": round(elapsed_time, 5),
    }
    return tokens, stats
