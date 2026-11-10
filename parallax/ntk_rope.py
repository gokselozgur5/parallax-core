"""NTK-aware Scaled Rotary Position Embedding (RoPE)."""

import numpy as np


class NTKAwareRoPE:
    """Computes modulated RoPE frequencies using NTK-aware base scaling."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        alpha: float = 1.0,
    ) -> None:
        self.dim = int(dim)
        self.base = float(base)
        self.alpha = float(alpha)

    def compute_scaled_inv_freq(self) -> np.ndarray:
        """Compute NTK-adjusted inverse frequencies."""
        scaled_base = self.base * (self.alpha ** (self.dim / (self.dim - 2)))
        indices = np.arange(0, self.dim, 2).astype(np.float64)
        return 1.0 / (scaled_base ** (indices / self.dim))

    def apply_rope(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply NTK-aware rotary embeddings to input tensor x."""
        inv_freq = self.compute_scaled_inv_freq()
        t = np.arange(seq_len, dtype=np.float64)
        freqs = np.outer(t, inv_freq)
        emb = np.repeat(freqs, 2, axis=-1)
        cos, sin = np.cos(emb), np.sin(emb)

        # Rotate half: [-x2, x1, -x4, x3, ...]
        x_rot = np.zeros_like(x)
        x_rot[..., 0::2] = -x[..., 1::2]
        x_rot[..., 1::2] = x[..., 0::2]
        return x * cos + x_rot * sin
