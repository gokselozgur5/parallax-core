"""DeepSeek Multi-Head Latent Attention (MLA) implementation."""

from typing import Optional, Tuple
import numpy as np
from parallax.attention import apply_rope


class MultiHeadLatentAttention:
    """DeepSeek Multi-Head Latent Attention (MLA) with decoupled RoPE and latent KV cache."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        rope_dim: int = 16,
        theta: float = 10000.0,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.theta = float(theta)
        self.head_dim = embed_dim // num_heads

        scale = 1.0 / np.sqrt(self.embed_dim)
        # Latent KV down-projection and up-projections
        self.w_dkv = np.random.randn(self.embed_dim, self.latent_dim) * scale
        self.w_uk = np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale
        self.w_uv = np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale

        # Decoupled RoPE key projection
        self.w_kr = np.random.randn(self.embed_dim, self.rope_dim) * scale

        # Query projection (content heads + decoupled rope head)
        self.w_qc = np.random.randn(self.embed_dim, self.num_heads * self.head_dim) * scale
        self.w_qr = np.random.randn(self.embed_dim, self.num_heads * self.rope_dim) * scale

        # Output projection
        self.w_o = np.random.randn(self.num_heads * self.head_dim, self.embed_dim) * scale

    @property
    def w_q(self) -> np.ndarray:
        """Backward compatibility alias for content query weights."""
        return self.w_qc

    @w_q.setter
    def w_q(self, val: np.ndarray) -> None:
        self.w_qc = val

    def compress_kv(self, x: np.ndarray) -> np.ndarray:
        """Compress input x into latent KV representation c_t^KV."""
        return np.matmul(x, self.w_dkv)

    def decompress_kv(self, c_kv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project latent vector back to distributed key and value matrices."""
        k = np.matmul(c_kv, self.w_uk)
        v = np.matmul(c_kv, self.w_uv)
        return k, v

    def forward(
        self,
        x: np.ndarray,
        causal: bool = True,
        past_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_cache: bool = False,
    ):
        """Compute multi-head latent attention output with decoupled RoPE and latent KV cache."""
        was_2d = (x.ndim == 2)
        if was_2d:
            x = x[None, :, :]
        B, T, _ = x.shape
        past_len = past_kv[0].shape[1] if past_kv is not None else 0

        # Latent KV compression and decoupled key RoPE
        c_kv_step = self.compress_kv(x)
        k_r_step = np.matmul(x, self.w_kr)
        k_r_rot_step = apply_rope(
            k_r_step, seq_dim=1, offset=past_len, theta=self.theta, interleaved=True
        )

        if past_kv is not None:
            c_kv = np.concatenate([past_kv[0], c_kv_step], axis=1)
            k_r_rot = np.concatenate([past_kv[1], k_r_rot_step], axis=1)
        else:
            c_kv = c_kv_step
            k_r_rot = k_r_rot_step

        new_kv = (c_kv, k_r_rot) if use_cache else None

        # Content and decoupled rotary query heads
        q_c = np.matmul(x, self.w_qc).reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        q_r = np.matmul(x, self.w_qr).reshape(B, T, self.num_heads, self.rope_dim)
        q_r_rot = apply_rope(
            q_r, seq_dim=1, offset=past_len, theta=self.theta, interleaved=True
        ).swapaxes(1, 2)

        # Decompress cached latent KV into content heads
        S = c_kv.shape[1]
        k_c, v_c = self.decompress_kv(c_kv)
        k_c = k_c.reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        v_c = v_c.reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        k_r_rot_heads = np.expand_dims(k_r_rot, axis=1)

        # Attention scores combining content and decoupled RoPE keys
        scores_c = np.matmul(q_c, k_c.swapaxes(-1, -2))
        scores_r = np.matmul(q_r_rot, k_r_rot_heads.swapaxes(-1, -2))
        scale = 1.0 / np.sqrt(self.head_dim + self.rope_dim)
        scores = (scores_c + scores_r) * scale

        if causal:
            q_pos = np.arange(T)[:, None] + past_len
            k_pos = np.arange(S)[None, :]
            causal_mask = k_pos <= q_pos
            scores = np.where(causal_mask[None, None, :, :], scores, -np.inf)

        max_scores = np.max(scores, axis=-1, keepdims=True)
        all_masked = np.isneginf(max_scores)
        safe_max = np.where(all_masked, 0.0, max_scores)
        exp_s = np.exp(scores - safe_max)
        exp_s = np.where(all_masked, 0.0, exp_s)
        sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
        weights = exp_s / np.where(sum_exp == 0.0, 1.0, sum_exp)
        context = np.matmul(weights, v_c).swapaxes(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = np.matmul(context, self.w_o)

        if was_2d:
            out = out.squeeze(0)
        if use_cache:
            return out, new_kv
        return out


from parallax.nn import Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


class MLAModule(Module):
    """Autograd-integrated Module adapter for DeepSeek Multi-Head Latent Attention.

    Can be dropped into TransformerBlock / ParallaxLM models.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        latent_dim: int,
        rope_dim: int = 16,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.latent_dim = int(latent_dim)
        self.rope_dim = int(rope_dim)
        self.theta = float(theta)
        self.head_dim = embed_dim // num_heads

        scale = 1.0 / np.sqrt(self.embed_dim)
        self.w_dkv = Parameter(np.random.randn(self.embed_dim, self.latent_dim) * scale, label="MLA.w_dkv")
        self.w_uk = Parameter(np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale, label="MLA.w_uk")
        self.w_uv = Parameter(np.random.randn(self.latent_dim, self.num_heads * self.head_dim) * scale, label="MLA.w_uv")
        self.w_kr = Parameter(np.random.randn(self.embed_dim, self.rope_dim) * scale, label="MLA.w_kr")
        self.w_qc = Parameter(np.random.randn(self.embed_dim, self.num_heads * self.head_dim) * scale, label="MLA.w_qc")
        self.w_qr = Parameter(np.random.randn(self.embed_dim, self.num_heads * self.rope_dim) * scale, label="MLA.w_qr")
        self.w_o = Parameter(np.random.randn(self.num_heads * self.head_dim, self.embed_dim) * scale, label="MLA.w_o")

    def forward(
        self,
        x: Tensor,
        causal: bool = True,
        past_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_cache: bool = False,
    ):
        was_2d = (x.data.ndim == 2)
        x_in = x.data if not was_2d else x.data[None, :, :]
        B, T, _ = x_in.shape
        past_len = past_kv[0].shape[1] if past_kv is not None else 0

        c_kv_step = np.matmul(x_in, self.w_dkv.data)
        k_r_step = np.matmul(x_in, self.w_kr.data)
        k_r_rot_step = apply_rope(
            k_r_step, seq_dim=1, offset=past_len, theta=self.theta, interleaved=True
        )

        if past_kv is not None:
            c_kv = np.concatenate([past_kv[0], c_kv_step], axis=1)
            k_r_rot = np.concatenate([past_kv[1], k_r_rot_step], axis=1)
        else:
            c_kv = c_kv_step
            k_r_rot = k_r_rot_step

        new_kv = (c_kv, k_r_rot) if use_cache else None

        q_c = np.matmul(x_in, self.w_qc.data).reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
        q_r = np.matmul(x_in, self.w_qr.data).reshape(B, T, self.num_heads, self.rope_dim)
        q_r_rot = apply_rope(
            q_r, seq_dim=1, offset=past_len, theta=self.theta, interleaved=True
        ).swapaxes(1, 2)

        S = c_kv.shape[1]
        k_c = np.matmul(c_kv, self.w_uk.data).reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        v_c = np.matmul(c_kv, self.w_uv.data).reshape(B, S, self.num_heads, self.head_dim).swapaxes(1, 2)
        k_r_rot_heads = np.expand_dims(k_r_rot, axis=1)

        scores_c = np.matmul(q_c, k_c.swapaxes(-1, -2))
        scores_r = np.matmul(q_r_rot, k_r_rot_heads.swapaxes(-1, -2))
        scale = 1.0 / np.sqrt(self.head_dim + self.rope_dim)
        scores = (scores_c + scores_r) * scale

        if causal:
            q_pos = np.arange(T)[:, None] + past_len
            k_pos = np.arange(S)[None, :]
            causal_mask = k_pos <= q_pos
            scores = np.where(causal_mask[None, None, :, :], scores, -np.inf)

        max_scores = np.max(scores, axis=-1, keepdims=True)
        all_masked = np.isneginf(max_scores)
        safe_max = np.where(all_masked, 0.0, max_scores)
        exp_s = np.exp(scores - safe_max)
        exp_s = np.where(all_masked, 0.0, exp_s)
        sum_exp = np.sum(exp_s, axis=-1, keepdims=True)
        weights = exp_s / np.where(sum_exp == 0.0, 1.0, sum_exp)

        context = np.matmul(weights, v_c).swapaxes(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out_data = np.matmul(context, self.w_o.data)

        if was_2d:
            out_data = out_data.squeeze(0)

        params = (x, self.w_dkv, self.w_uk, self.w_uv, self.w_kr, self.w_qc, self.w_qr, self.w_o)
        out_tensor = Tensor(out_data, params, "MLAModule")

        if not out_tensor.requires_grad:
            return (out_tensor, new_kv) if use_cache else out_tensor

        def _backward() -> None:
            if out_tensor.grad is None:
                return
            grad_y = out_tensor.grad if not was_2d else out_tensor.grad[None, :, :]
            grad_y_2d = grad_y.reshape(-1, self.embed_dim)
            context_2d = context.reshape(-1, self.num_heads * self.head_dim)

            if self.w_o.requires_grad:
                grad_wo = np.matmul(context_2d.T, grad_y_2d)
                _accumulate_grad(self.w_o, grad_wo)

            grad_context = np.matmul(grad_y, self.w_o.data.T).reshape(B, T, self.num_heads, self.head_dim).swapaxes(1, 2)
            grad_weights = np.matmul(grad_context, v_c.swapaxes(-1, -2))
            grad_scores = weights * (grad_weights - np.sum(grad_weights * weights, axis=-1, keepdims=True)) * scale

            grad_qc = np.matmul(grad_scores, k_c)
            grad_qc_flat = grad_qc.swapaxes(1, 2).reshape(B * T, self.num_heads * self.head_dim)
            if self.w_qc.requires_grad:
                x_flat = x_in.reshape(B * T, self.embed_dim)
                _accumulate_grad(self.w_qc, np.matmul(x_flat.T, grad_qc_flat))

            if x.requires_grad:
                grad_x = np.matmul(grad_qc_flat, self.w_qc.data.T).reshape(x.data.shape)
                _accumulate_grad(x, grad_x)

        out_tensor._backward = _backward
        return (out_tensor, new_kv) if use_cache else out_tensor
