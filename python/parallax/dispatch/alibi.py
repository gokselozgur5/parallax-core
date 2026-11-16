"""ALiBi (Attention with Linear Biases) positional bias dispatch helpers."""

from __future__ import annotations

import numpy as np


def get_alibi_slopes(num_heads: int) -> np.ndarray:
    """Compute ALiBi head slopes following Press et al. (2021)."""
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    def _slopes(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start * (start**i) for i in range(n)]

    if (num_heads & (num_heads - 1)) == 0:
        return np.array(_slopes(num_heads), dtype=np.float32)
    n = 2 ** int(np.floor(np.log2(num_heads)))
    base = _slopes(n)
    extra = _slopes(2 * n)[0::2][: num_heads - n]
    return np.array(base + extra, dtype=np.float32)
