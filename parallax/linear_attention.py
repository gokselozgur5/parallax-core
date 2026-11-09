"""
Linear Causal Attention with Recurrent Hidden State Formulation.

Implements O(N) linear attention using the kernel feature map:
phi(x) = elu(x) + 1.0 (Katharopoulos et al., 2020).
Provides both parallel causal prefix-sum training forward and recurrent
step forward for constant O(1) memory autoregressive decoding.
"""

from typing import Optional, Tuple, Union
import numpy as np
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor


def elu_feature_map(x: np.ndarray) -> np.ndarray:
    """Non-negative kernel feature map phi(x) = elu(x) + 1."""
    return np.where(x > 0, x + 1.0, np.exp(x))


class LinearAttention(Module):
    """Linear Causal Self-Attention with dual parallel/recurrent forward paths."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps

        self.q_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=qkv_bias)

    def forward(
        self,
        x: Union[Tensor, np.ndarray],
        state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[np.ndarray, np.ndarray]]]:
        """Forward pass supporting parallel prefix sum or recurrent single-step.

        Args:
            x: Input tensor [B, T, C]
            state: Optional recurrent state tuple (S, Z) where:
                   S is [B, num_heads, head_dim, head_dim]
                   Z is [B, num_heads, head_dim, 1]
        """
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        B, T, C = x_tensor.shape

        q = self.q_proj(x_tensor).data.reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_proj(x_tensor).data.reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        v = self.v_proj(x_tensor).data.reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)

        # Feature maps: phi(Q), phi(K) -> [B, num_heads, T, head_dim]
        phi_q = elu_feature_map(q)
        phi_k = elu_feature_map(k)

        if state is not None or T == 1:
            # Recurrent O(1) step update per token
            if state is None:
                prev_S = np.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=q.dtype)
                prev_Z = np.zeros((B, self.num_heads, self.head_dim, 1), dtype=q.dtype)
            else:
                prev_S, prev_Z = state

            # Single token step T=1
            q_t = phi_q[:, :, 0:1, :]  # [B, H, 1, D]
            k_t = phi_k[:, :, 0:1, :]  # [B, H, 1, D]
            v_t = v[:, :, 0:1, :]      # [B, H, 1, D]

            # S_new = S_prev + K_t^T @ V_t -> [B, H, D, D]
            # Z_new = Z_prev + K_t^T       -> [B, H, D, 1]
            k_t_col = np.swapaxes(k_t, -1, -2)  # [B, H, D, 1]
            new_S = prev_S + (k_t_col @ v_t)
            new_Z = prev_Z + k_t_col

            # num = Q_t @ S_new -> [B, H, 1, D]
            # den = Q_t @ Z_new -> [B, H, 1, 1]
            num = q_t @ new_S
            den = (q_t @ new_Z) + self.eps
            out_heads = num / den

            merged = out_heads.swapaxes(1, 2).reshape(B, 1, C)
            out = self.out_proj(Tensor(merged, (x_tensor,), "LinearAttention"))
            if state is not None:
                return out, (new_S, new_Z)
            return out

        # Parallel prefix-sum training forward for T > 1
        # KV prefix product: K_t^T @ V_t for each t
        # Shape phi_k: [B, H, T, D], v: [B, H, T, D]
        kv = np.matmul(np.swapaxes(phi_k, -1, -2)[..., :, :, None], v[..., :, None, :])
        # kv shape: [B, H, T, D, D]
        S_cum = np.cumsum(kv, axis=2)  # [B, H, T, D, D]
        Z_cum = np.cumsum(np.swapaxes(phi_k, -1, -2)[..., :, :, None], axis=2)  # [B, H, T, D, 1]

        # Q_t @ S_cum_t -> [B, H, T, 1, D] -> [B, H, T, D]
        q_expanded = phi_q[..., :, None, :]  # [B, H, T, 1, D]
        num = np.matmul(q_expanded, S_cum).squeeze(-2)  # [B, H, T, D]
        den = np.matmul(q_expanded, Z_cum).squeeze(-2) + self.eps  # [B, H, T, 1]

        out_heads = num / den  # [B, H, T, D]
        merged = out_heads.swapaxes(1, 2).reshape(B, T, C)
        out = self.out_proj(Tensor(merged, (x_tensor,), "LinearAttention"))
        return out


