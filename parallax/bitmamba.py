"""
BitMamba: Zero-RAM / Sub-2-bit Selective State Space Model Architecture.

Fuses BitNet b1.58 ternary weights {-1, 0, +1} and INT8 activation quantization
with Selective State Space (Mamba / S6) recurrence. Eliminates KV cache for O(1) memory
inference while replacing energy-intensive floating-point multiplications with integer
addition-dominated matrix operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from parallax.bitnet import BitLinear
from parallax.mamba import (
    causal_conv1d_backward,
    causal_conv1d_forward,
    selective_scan_backward,
    selective_scan_forward,
)
from parallax.models.gpt import Embedding
from parallax.nn import Module, Parameter, RMSNorm
from parallax.tensor import Tensor, _accumulate_grad


class BitMambaBlock(Module):
    """Selective State Space (Mamba) Block powered by 1.58-bit BitLinear projections."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, d_model // 16)

        # 1.58-bit Ternary In-projection for x and gate branches [d_model -> 2 * d_inner]
        self.in_proj = BitLinear(d_model, 2 * self.d_inner, bias=False)

        # 1D Depthwise causal convolution weights: [d_inner, d_conv]
        self.conv_weight = Parameter(np.random.randn(self.d_inner, d_conv) * 0.02, label="BitMambaBlock.conv_weight")
        self.conv_bias = Parameter(np.zeros(self.d_inner), label="BitMambaBlock.conv_bias")

        # 1.58-bit Ternary Input-dependent projections: x -> (delta, B, C)
        self.x_proj = BitLinear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = BitLinear(self.dt_rank, self.d_inner, bias=True)

        # Continuous transition parameters for HiPPO stability
        A_init = np.repeat(np.arange(1, d_state + 1, dtype=np.float64)[None, :], self.d_inner, axis=0)
        self.A_log = Parameter(np.log(A_init), label="BitMambaBlock.A_log")
        self.D = Parameter(np.ones(self.d_inner), label="BitMambaBlock.D")

        # 1.58-bit Ternary Out-projection back to d_model
        self.out_proj = BitLinear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        u: Tensor,
        ssm_state: Optional[np.ndarray] = None,
        conv_state: Optional[np.ndarray] = None,
    ) -> Union[Tensor, Tuple[Tensor, np.ndarray, np.ndarray]]:
        """Forward pass of BitMambaBlock with exact analytical autograd reverse-mode graph.

        Supports step-by-step autoregressive O(1) state caching when ssm_state and conv_state are provided.
        """
        B, T, D = u.shape

        # Step mode (T == 1 with states provided)
        if ssm_state is not None and conv_state is not None:
            # 1. Dual projection
            xz = self.in_proj(u)  # [B, 1, 2 * d_inner]
            x_data = xz.data[:, 0, : self.d_inner]  # [B, d_inner]
            z_data = xz.data[:, 0, self.d_inner :]  # [B, d_inner]

            # Update conv state: shift left and append current x
            # conv_state: [B, d_inner, d_conv]
            new_conv_state = np.zeros_like(conv_state)
            new_conv_state[:, :, :-1] = conv_state[:, :, 1:]
            new_conv_state[:, :, -1] = x_data

            # 1D conv over kernel window
            x_conv = np.sum(new_conv_state * self.conv_weight.data[None, :, :], axis=-1) + self.conv_bias.data
            sig_conv = 1.0 / (1.0 + np.exp(-np.clip(x_conv, -50.0, 50.0)))
            x_conv = x_conv * sig_conv  # [B, d_inner]

            # x_proj
            x_conv_tensor = Tensor(x_conv[:, None, :])
            x_dbl = self.x_proj(x_conv_tensor)
            dt_data = x_dbl.data[:, 0, : self.dt_rank]
            B_ssm = x_dbl.data[:, 0, self.dt_rank : self.dt_rank + self.d_state]
            C_ssm = x_dbl.data[:, 0, self.dt_rank + self.d_state :]

            dt_tensor = Tensor(dt_data[:, None, :])
            dt_proj_out = self.dt_proj(dt_tensor)
            delta = np.log1p(np.exp(np.clip(dt_proj_out.data[:, 0, :], -50.0, 50.0)))

            # SSM Discretization ZOH
            A = -np.exp(self.A_log.data)  # [d_inner, d_state]
            # A_bar: [B, d_inner, d_state]
            A_bar = np.exp(delta[:, :, None] * A[None, :, :])
            B_bar = delta[:, :, None] * B_ssm[:, None, :]

            # Recurrence step: h_new = A_bar * h_prev + B_bar * x_conv
            new_ssm_state = A_bar * ssm_state + B_bar * x_conv[:, :, None]
            # y = sum_n (C_ssm * h_new) + D * x_conv
            y = np.sum(C_ssm[:, None, :] * new_ssm_state, axis=-1) + self.D.data[None, :] * x_conv

            # Gate modulation
            z_sig = 1.0 / (1.0 + np.exp(-np.clip(z_data, -50.0, 50.0)))
            y_gated = y * (z_data * z_sig)

            y_tensor = Tensor(y_gated[:, None, :])
            out = self.out_proj(y_tensor)
            return out, new_ssm_state, new_conv_state

        # 1. Dual projection into branch x and branch z (gate) via BitLinear
        xz = self.in_proj(u)  # [B, T, 2 * d_inner]
        x_data = xz.data[:, :, : self.d_inner]
        z_data = xz.data[:, :, self.d_inner :]

        # 2. Causal 1D Convolution along sequence length
        conv_out = causal_conv1d_forward(x_data, self.conv_weight.data, self.conv_bias.data)

        # SiLU activation: x_conv = conv_out * sigmoid(conv_out)
        sig_conv = 1.0 / (1.0 + np.exp(-np.clip(conv_out, -50.0, 50.0)))
        x_conv = conv_out * sig_conv

        # 3. Input-dependent parameter discretization (delta, B, C) via BitLinear
        x_conv_tensor = Tensor(x_conv, requires_grad=True)
        x_dbl = self.x_proj(x_conv_tensor)  # [B, T, dt_rank + 2 * d_state]
        dt_data = x_dbl.data[:, :, : self.dt_rank]
        B_ssm_data = x_dbl.data[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm_data = x_dbl.data[:, :, self.dt_rank + self.d_state :]

        # Delta projection via BitLinear with softplus: delta = log(1 + exp(dt_proj(dt)))
        dt_tensor = Tensor(dt_data, requires_grad=True)
        dt_proj_out = self.dt_proj(dt_tensor)
        delta_sig = 1.0 / (1.0 + np.exp(-np.clip(dt_proj_out.data, -50.0, 50.0)))
        delta = np.log1p(np.exp(np.clip(dt_proj_out.data, -50.0, 50.0)))

        # 4. Selective Scan Recurrence
        A = -np.exp(self.A_log.data)
        y, h_states = selective_scan_forward(x_conv, delta, A, B_ssm_data, C_ssm_data, D=self.D.data)

        # 5. SiLU Gate Modulation: y_gated = y * SiLU(z)
        z_sig = 1.0 / (1.0 + np.exp(-np.clip(z_data, -50.0, 50.0)))
        silu_z = z_data * z_sig
        y_gated = y * silu_z

        # 6. Autograd wrapper connecting upstream DAG
        children = (xz, self.conv_weight, self.conv_bias, self.A_log, self.D)
        y_gated_tensor = Tensor(y_gated, children, "BitMambaScan")

        def _backward() -> None:
            if y_gated_tensor.grad is None:
                return

            d_y_gated = y_gated_tensor.grad

            # 1. Gate backward
            d_y = d_y_gated * silu_z
            d_silu_z = d_y_gated * y
            d_z = d_silu_z * (z_sig * (1.0 + z_data * (1.0 - z_sig)))

            # 2. Selective scan backward
            du_ssm, ddelta, dA, dB_ssm, dC_ssm, dD = selective_scan_backward(
                d_y, x_conv, delta, A, B_ssm_data, C_ssm_data, self.D.data, h_states
            )

            if self.D.requires_grad and dD is not None:
                _accumulate_grad(self.D, dD)

            if self.A_log.requires_grad:
                _accumulate_grad(self.A_log, dA * A)

            # 3. Delta softplus & dt_proj backward
            d_dt_proj_out = ddelta * delta_sig
            dt_proj_out.backward(d_dt_proj_out)

            # 4. x_dbl backward
            d_x_dbl = np.zeros_like(x_dbl.data)
            if dt_tensor.grad is not None:
                d_x_dbl[:, :, : self.dt_rank] = dt_tensor.grad
            d_x_dbl[:, :, self.dt_rank : self.dt_rank + self.d_state] = dB_ssm
            d_x_dbl[:, :, self.dt_rank + self.d_state :] = dC_ssm

            x_dbl.backward(d_x_dbl)

            # 5. SiLU backward
            d_x_conv = du_ssm + (x_conv_tensor.grad if x_conv_tensor.grad is not None else 0.0)
            d_conv_out = d_x_conv * (sig_conv * (1.0 + conv_out * (1.0 - sig_conv)))

            # 6. Causal 1D Conv backward
            dx, d_conv_w, d_conv_b = causal_conv1d_backward(
                d_conv_out, x_data, self.conv_weight.data, self.conv_bias.data
            )

            if self.conv_weight.requires_grad:
                _accumulate_grad(self.conv_weight, d_conv_w)

            if self.conv_bias.requires_grad:
                _accumulate_grad(self.conv_bias, d_conv_b)

            # 7. xz projection split backward through in_proj
            d_xz = np.concatenate([dx, d_z], axis=-1)
            xz.backward(d_xz)

        y_gated_tensor._backward = _backward
        return self.out_proj(y_gated_tensor)


@dataclass
class BitMambaConfig:
    """Configuration for BitMamba Language Model."""

    vocab_size: int = 256
    d_model: int = 64
    num_layers: int = 3
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    rms_norm_eps: float = 1e-6
    tie_weights: bool = False


class BitMambaLM(Module):
    """Full Autoregressive Language Model combining BitNet b1.58 quantization with Mamba SSM.

    Achieves O(1) state memory during inference without KV caches, while replacing
    weight-activation multiplications with ternary additions across all linear projections.
    """

    def __init__(self, config: BitMambaConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.blocks = [
            BitMambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        # 1.58-bit LM Head
        self.lm_head = BitLinear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        idx: Union[np.ndarray, Tensor],
        states: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        return_hidden: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tuple[np.ndarray, np.ndarray]]], Tuple[Tensor, Tensor]]:
        """Forward pass over sequence or single step.

        Parameters
        ----------
        idx : array_like or Tensor of token indices [B, T]
        states : optional list of (ssm_state, conv_state) per layer for O(1) memory step-by-step decoding.
        return_hidden : whether to return final hidden activations before LM head.
        """
        x = self.token_embeddings(idx)

        new_states: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, block in enumerate(self.blocks):
            if states is not None:
                ssm_s, conv_s = states[i]
                x, new_ssm_s, new_conv_s = block(x, ssm_state=ssm_s, conv_state=conv_s)
                new_states.append((new_ssm_s, new_conv_s))
            else:
                x = block(x)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        if states is not None:
            return logits, new_states
        return logits
