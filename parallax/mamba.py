"""
Selective State Space Model (S6 / Mamba: Gu & Dao, 2023) mathematical primitives.
"""

from typing import Optional, Tuple
import numpy as np
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad


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


def causal_conv1d_forward(
    x: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Causal depthwise 1D convolution along sequence dimension."""
    B, T, d_inner = x.shape
    d_conv = weight.shape[1]
    conv_out = np.zeros_like(x)
    for t in range(T):
        t_start = max(0, t - d_conv + 1)
        w_len = t - t_start + 1
        window = x[:, t_start : t + 1, :]
        w_slice = weight[:, -w_len:]
        conv_out[:, t, :] = np.sum(window * w_slice.T[None, :, :], axis=1) + bias
    return conv_out


def causal_conv1d_backward(
    d_out: np.ndarray, x: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical reverse-mode gradient of causal depthwise 1D convolution."""
    B, T, d_inner = x.shape
    d_conv = weight.shape[1]

    d_x = np.zeros_like(x)
    d_weight = np.zeros_like(weight)
    d_bias = np.sum(d_out, axis=(0, 1))

    for t in range(T):
        t_start = max(0, t - d_conv + 1)
        w_len = t - t_start + 1
        window = x[:, t_start : t + 1, :]
        grad_t = d_out[:, t, :]

        dw_slice = np.sum(window * grad_t[:, None, :], axis=0).T
        d_weight[:, -w_len:] += dw_slice

        w_slice = weight[:, -w_len:]
        d_window = grad_t[:, None, :] * w_slice.T[None, :, :]
        d_x[:, t_start : t + 1, :] += d_window

    return d_x, d_weight, d_bias


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
        """Forward pass of MambaBlock with exact analytical autograd reverse-mode graph."""
        B, T, D = u.shape

        # 1. Dual projection into branch x and branch z (gate)
        xz = self.in_proj(u)  # [B, T, 2 * d_inner]
        x_data = xz.data[:, :, : self.d_inner]
        z_data = xz.data[:, :, self.d_inner :]

        # 2. Causal 1D Convolution along sequence length
        conv_out = causal_conv1d_forward(x_data, self.conv_weight.data, self.conv_bias.data)

        # SiLU activation: x_conv = conv_out * sigmoid(conv_out)
        sig_conv = 1.0 / (1.0 + np.exp(-np.clip(conv_out, -50.0, 50.0)))
        x_conv = conv_out * sig_conv

        # 3. Input-dependent parameter discretization (delta, B, C)
        x_conv_tensor = Tensor(x_conv, requires_grad=True)
        x_dbl = self.x_proj(x_conv_tensor)  # [B, T, dt_rank + 2 * d_state]
        dt_data = x_dbl.data[:, :, : self.dt_rank]
        B_ssm_data = x_dbl.data[:, :, self.dt_rank : self.dt_rank + self.d_state]
        C_ssm_data = x_dbl.data[:, :, self.dt_rank + self.d_state :]

        # Delta projection with softplus: delta = log(1 + exp(dt_proj(dt)))
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
        y_gated_tensor = Tensor(y_gated, children, "MambaScan")

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
                # A = -exp(A_log) -> dA_log = dA * (-exp(A_log)) = dA * A
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

            # 7. xz projection split backward
            d_xz = np.concatenate([dx, d_z], axis=-1)
            if xz.requires_grad:
                _accumulate_grad(xz, d_xz)

        y_gated_tensor._backward = _backward
        return self.out_proj(y_gated_tensor)



