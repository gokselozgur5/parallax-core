"""
Perplexity and Language Model Evaluation Suite.

Provides cross-entropy evaluation and sliding-window perplexity (PPL)
computation for autoregressive models without external dependencies.
"""

from typing import List, Optional
import numpy as np
from parallax.models.gpt import ParallaxLM
from parallax.tensor import Tensor


def compute_cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """Computes cross-entropy loss from unnormalized logits and integer targets.

    Args:
        logits: Array of shape [N, vocab_size]
        targets: 1D array of shape [N] with ground-truth integer token IDs.

    Returns:
        Mean negative log-likelihood across tokens.
    """
    if logits.ndim != 2:
        raise ValueError("logits must be 2D [N, vocab_size]")
    if targets.ndim != 1:
        raise ValueError("targets must be 1D [N]")
    if len(logits) != len(targets):
        raise ValueError("logits and targets length mismatch")

    # Numerically stable log-softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = (logits - max_logits) - np.log(sum_exp)

    # Pick log prob of true targets
    n_tokens = len(targets)
    target_log_probs = log_probs[np.arange(n_tokens), targets]
    return float(-np.mean(target_log_probs))


def compute_perplexity(
    model: ParallaxLM,
    tokens: List[int],
    max_context_len: Optional[int] = None,
    stride: Optional[int] = None,
) -> float:
    """Computes sliding-window perplexity (PPL) over a sequence of token IDs.

    Args:
        model: ParallaxLM language model.
        tokens: Full sequence of token IDs.
        max_context_len: Maximum chunk size evaluated at once (defaults to model's max_seq_len).
        stride: Stride between successive evaluation windows (defaults to max_context_len // 2).

    Returns:
        Perplexity value (exp of average negative log-likelihood per token).
    """
    seq_len = len(tokens)
    if seq_len < 2:
        raise ValueError("At least 2 tokens are required to evaluate perplexity")

    window_size = max_context_len or model.config.max_seq_len
    window_stride = stride or max(1, window_size // 2)

    total_nll = 0.0
    total_tokens_evaluated = 0

    tokens_arr = np.array(tokens, dtype=np.int64)

    for start_idx in range(0, seq_len - 1, window_stride):
        end_idx = min(start_idx + window_size, seq_len)
        chunk = tokens_arr[start_idx:end_idx]
        if len(chunk) < 2:
            break

        # Input is chunk[:-1], target is chunk[1:]
        inputs = chunk[:-1][None, :]  # [1, T]
        targets = chunk[1:]           # [T]

        out = model(inputs)
        logits = out.data if isinstance(out, Tensor) else out
        logits_2d = logits[0]  # [T, vocab_size]

        # Determine which target positions are accounted for by this stride
        if start_idx == 0:
            target_mask_start = 0
        else:
            # Skip targets already scored in previous stride window
            target_mask_start = window_size - 1 - window_stride
            target_mask_start = max(0, min(target_mask_start, len(targets)))

        eval_targets = targets[target_mask_start:]
        eval_logits = logits_2d[target_mask_start:]

        if len(eval_targets) == 0:
            continue

        n_chunk_tokens = len(eval_targets)
        chunk_loss = compute_cross_entropy_loss(eval_logits, eval_targets)
        total_nll += chunk_loss * n_chunk_tokens
        total_tokens_evaluated += n_chunk_tokens

        if end_idx == seq_len:
            break

    if total_tokens_evaluated == 0:
        raise RuntimeError("No tokens were evaluated for perplexity")

    mean_nll = total_nll / total_tokens_evaluated
    return float(np.exp(mean_nll))

