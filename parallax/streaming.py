"""
Sliding Window Attention with Attention Sinks (StreamingLLM: Xiao et al., 2023).
"""

import numpy as np


def generate_sink_window_mask(
    seq_len: int,
    window_size: int = 16,
    num_sinks: int = 4,
) -> np.ndarray:
    """Constructs a boolean attention mask for StreamingLLM with sink tokens."""
    q_idx = np.arange(seq_len)[:, None]
    k_idx = np.arange(seq_len)[None, :]

    causal_mask = q_idx >= k_idx
    sink_mask = k_idx < num_sinks
    window_mask = (q_idx - k_idx) < window_size

    # Valid if causal AND (is a sink token OR within sliding window)
    valid = causal_mask & (sink_mask | window_mask)
    return valid
