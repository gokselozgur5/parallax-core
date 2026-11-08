"""
Kahneman-Tversky Optimization (KTO: Ethayarajh et al., 2024) from first principles.
"""

from typing import Optional, Tuple
import numpy as np
from parallax.rl import compute_sequence_log_probs
from parallax.tensor import Tensor


def kto_value_function(
    z: np.ndarray,
    is_desirable: bool = True,
    lambda_d: float = 1.0,
    lambda_u: float = 1.33,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prospect Theory value function v(z) and its analytical derivative dv/dz."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    if is_desirable:
        # v(z) = 1 - sigma(z) = sigma(-z)
        loss = lambda_d * (1.0 - sig)
        # d(loss)/dz = - lambda_d * sig * (1 - sig)
        dloss_dz = - lambda_d * sig * (1.0 - sig)
    else:
        # v(z) = 1 - sigma(-z) = sigma(z)
        loss = lambda_u * sig
        # d(loss)/dz = lambda_u * sig * (1 - sig)
        dloss_dz = lambda_u * sig * (1.0 - sig)

    return loss, dloss_dz


def compute_kto_implicit_rewards(
    policy_logits: Tensor,
    ref_logits: np.ndarray,
    targets: np.ndarray,
    beta: float = 0.1,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes implicit rewards r_theta(x, y) = beta * (log pi(y|x) - log ref(y|x)) and policy logit grads."""
    pi_logp, d_pi = compute_sequence_log_probs(policy_logits, targets, mask)
    ref_tensor = Tensor(ref_logits, requires_grad=False)
    ref_logp, _ = compute_sequence_log_probs(ref_tensor, targets, mask)

    log_ratio = pi_logp - ref_logp
    implicit_rewards = beta * log_ratio
    return implicit_rewards, d_pi


def kto_loss(
    policy_logits: Tensor,
    ref_logits: np.ndarray,
    targets: np.ndarray,
    is_desirable: bool = True,
    kl_ref_estimate: float = 0.0,
    beta: float = 0.1,
    lambda_d: float = 1.0,
    lambda_u: float = 1.33,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Computes Kahneman-Tversky Optimization (KTO) loss for unpaired binary feedback."""
    r_theta, d_pi = compute_kto_implicit_rewards(
        policy_logits, ref_logits, targets, beta=beta, mask=mask
    )

    # Reference-adjusted reward: z = r_theta - beta * KL_estimate
    z = r_theta - beta * kl_ref_estimate

    # Prospect Theory value loss
    loss_arr, dloss_dz = kto_value_function(
        z, is_desirable=is_desirable, lambda_d=lambda_d, lambda_u=lambda_u
    )
    loss_val = float(np.mean(loss_arr))

    # Analytical gradient backprop: dL / d(logits) = (dloss_dz * beta / B) * d_pi
    B = float(len(z))
    grad_scale = (dloss_dz * beta) / B  # Shape: [B]
    grad_logits = grad_scale[:, None, None] * d_pi

    if policy_logits.requires_grad:
        policy_logits.backward(grad=grad_logits)

    return loss_val, float(np.mean(r_theta))


