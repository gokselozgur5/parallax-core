"""
Selective State Space Model (S6 / Mamba: Gu & Dao, 2023) mathematical primitives.
"""

from typing import Optional, Tuple
import numpy as np
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor


def discretize_zoh(
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-Order Hold (ZOH) discretization for selective continuous-time state equations.

    A_bar = exp(delta * A)
    B_bar = delta * B
    """
    # delta: [B, T, D], A: [D, N] -> delta[:, :, :, None] * A[None, None, :, :]
    A_bar = np.exp(delta[:, :, :, None] * A[None, None, :, :])
    # B: [B, T, N] -> B_bar: [B, T, D, N]
    B_bar = delta[:, :, :, None] * B[:, :, None, :]
    return A_bar, B_bar


def selective_scan_forward(
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selective scan forward pass: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t, y_t = C_t * h_t + D * u_t."""
    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[1]

    A_bar, B_bar = discretize_zoh(delta, A, B)

    y = np.zeros((batch_size, seq_len, d_inner), dtype=np.float64)
    h_states = np.zeros((batch_size, seq_len, d_inner, d_state), dtype=np.float64)
    h_curr = np.zeros((batch_size, d_inner, d_state), dtype=np.float64)

    for t in range(seq_len):
        # h_curr = A_bar[:, t] * h_curr + B_bar[:, t] * u[:, t, :, None]
        h_curr = A_bar[:, t] * h_curr + B_bar[:, t] * u[:, t, :, None]
        h_states[:, t] = h_curr

        # y_t = sum_n (C_t[:, None, :] * h_curr) -> [B, D]
        y[:, t] = np.sum(C[:, t, None, :] * h_curr, axis=-1)

    if D is not None:
        y += u * D[None, None, :]

    return y, h_states


def selective_scan_backward(
    d_y: np.ndarray,
    u: np.ndarray,
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: Optional[np.ndarray],
    h_states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Analytical reverse-mode gradient of selective scan recurrence."""
    batch_size, seq_len, d_inner = u.shape
    d_state = A.shape[1]

    A_bar, B_bar = discretize_zoh(delta, A, B)

    d_u = np.zeros_like(u)
    d_delta = np.zeros_like(delta)
    d_A = np.zeros_like(A)
    d_B = np.zeros_like(B)
    d_C = np.zeros_like(C)
    d_D = np.sum(d_y * u, axis=(0, 1)) if D is not None else None

    if D is not None:
        d_u += d_y * D[None, None, :]

    d_h_future = np.zeros((batch_size, d_inner, d_state), dtype=np.float64)

    for t in reversed(range(seq_len)):
        d_C[:, t] = np.sum(d_y[:, t, :, None] * h_states[:, t], axis=1)

        d_h_from_y = d_y[:, t, :, None] * C[:, t, None, :]
        d_h_total = d_h_from_y + d_h_future

        d_u[:, t] += np.sum(d_h_total * B_bar[:, t], axis=-1)

        d_B_bar_t = d_h_total * u[:, t, :, None]
        d_B[:, t] += np.sum(d_B_bar_t * delta[:, t, :, None], axis=1)
        d_delta_from_B = np.sum(d_B_bar_t * B[:, t, None, :], axis=-1)

        if t > 0:
            h_prev = h_states[:, t - 1]
            d_A_bar_t = d_h_total * h_prev
            d_h_future = d_h_total * A_bar[:, t]
        else:
            d_A_bar_t = np.zeros_like(d_h_total)
            d_h_future = np.zeros_like(d_h_total)

        d_delta_from_A = np.sum(d_A_bar_t * A_bar[:, t] * A[None, :, :], axis=-1)
        d_delta[:, t] += d_delta_from_A + d_delta_from_B

        d_A += np.sum(d_A_bar_t * A_bar[:, t] * delta[:, t, :, None], axis=0)

    return d_u, d_delta, d_A, d_B, d_C, d_D


class MambaBlock(Module):
    """Selective State Space Model (Mamba) Decoder Block."""

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

        # In-projections for x and gate (x_proj & z_proj)
        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D Depthwise causal convolution weights: [d_inner, d_conv]
        self.conv_weight = Parameter(np.random.randn(self.d_inner, d_conv) * 0.02)
        self.conv_bias = Parameter(np.zeros(self.d_inner))

        # Input-dependent projections: x -> (delta, B, C)
        self.x_proj = Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = Linear(self.dt_rank, self.d_inner, bias=True)

        # Transition matrix A initialized as HiPPO / log-spaced diagonals (negated)
        A_init = np.repeat(np.arange(1, d_state + 1, dtype=np.float64)[None, :], self.d_inner, axis=0)
        self.A_log = Parameter(np.log(A_init))
        self.D = Parameter(np.ones(self.d_inner))

        # Out projection back to d_model
        self.out_proj = Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: Tensor) -> Tensor:
        """Forward pass of MambaBlock with linear selective scan."""
        B, T, D = u.shape

        # 1. Dual projection into branch x and branch z (gate)
        xz = self.in_proj(u)  # [B, T, 2 * d_inner]
        x_data = xz.data[:, :, : self.d_inner]
        z_data = xz.data[:, :, self.d_inner :]

        # 2. Causal 1D Convolution along sequence length
        conv_out = np.zeros_like(x_data)
        d_conv = self.conv_weight.data.shape[1]
        for t in range(T):
            t_start = max(0, t - d_conv + 1)
            window = x_data[:, t_start : t + 1, :]  # [B, w_len, d_inner]
            w_slice = self.conv_weight.data[:, -(t - t_start + 1) :]  # [d_inner, w_len]
            conv_out[:, t, :] = np.sum(window * w_slice.T[None, :, :], axis=1) + self.conv_bias.data

        # SiLU activation: x_conv = conv_out * sigmoid(conv_out)
        x_conv = conv_out * (1.0 / (1.0 + np.exp(-np.clip(conv_out, -50, 50))))

        # 3. Input-dependent parameter discretization (delta, B, C)
        x_conv_tensor = Tensor(x_conv)
        x_dbl = self.x_proj(x_conv_tensor).data  # [B, T, dt_rank + 2 * d_state]
        dt = x_dbl[:, :, : self.dt_rank]
        B_ssm = x_dbl[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm = x_dbl[:, :, self.dt_rank + self.d_state :]

        # Delta projection with softplus: delta = log(1 + exp(dt_proj(dt)))
        dt_proj_out = self.dt_proj(Tensor(dt)).data
        delta = np.log1p(np.exp(np.clip(dt_proj_out, -50, 50)))

        # 4. Selective Scan Recurrence
        A = -np.exp(self.A_log.data)
        y, _ = selective_scan_forward(x_conv, delta, A, B_ssm, C_ssm, D=self.D.data)

        # 5. SiLU Gate Modulation: y = y * SiLU(z)
        z_sig = 1.0 / (1.0 + np.exp(-np.clip(z_data, -50, 50)))
        y_gated = y * (z_data * z_sig)

        # 6. Out projection back to d_model
        y_gated_tensor = Tensor(y_gated, (u,), "MambaScan")

        def _backward() -> None:
            if y_gated_tensor.grad is not None and u.requires_grad:
                # Approximate residual pass-through gradient flow back to input
                if u.grad is not None:
                    u.grad += np.broadcast_to(y_gated_tensor.grad[:, :, :D], u.shape)

        y_gated_tensor._backward = _backward
        return self.out_proj(y_gated_tensor)



