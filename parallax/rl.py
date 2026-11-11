"""
First-principles Reward Modeling and Reinforcement Learning (RLHF / Policy Optimization).

Implements:
1. RewardModel: Language model backbone with a scalar head predicting alignment/human-preference scores
2. Bradley-Terry Preference Loss: Loss = -log(sigmoid(r_chosen - r_rejected))
3. REINFORCE / Policy Gradient Alignment Loop: Policy updates with reward feedback and KL regularization against reference model
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear, Module
from parallax.tensor import Tensor, _accumulate_grad


class RewardModel(Module):
    """Scalar Reward Model that outputs a scalar score r for any text sequence."""

    def __init__(self, config: ParallaxLMConfig):
        super().__init__()
        self.config = config
        self.backbone = ParallaxLM(config)
        # Linear projection from hidden dimension to single scalar reward
        self.reward_head = Linear(config.embed_dim, 1, bias=False)

    def forward(
        self,
        idx: np.ndarray,
        lengths: Optional[Union[np.ndarray, List[int]]] = None,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        # idx: [batch_size, seq_len]
        logits, hidden = self.backbone(idx, return_hidden=True)
        # Shape: [batch_size, seq_len, embed_dim]
        B, S, D = hidden.shape
        if lengths is not None:
            lengths_arr = np.asarray(lengths, dtype=np.int64)
            gather_indices = np.clip(lengths_arr - 1, 0, S - 1)
        elif attention_mask is not None:
            # attention_mask: [batch_size, seq_len]
            lengths_arr = np.asarray(np.sum(attention_mask, axis=-1), dtype=np.int64)
            gather_indices = np.clip(lengths_arr - 1, 0, S - 1)
        else:
            gather_indices = np.full(B, S - 1, dtype=np.int64)

        # Pool/select the last real token representation h_{lengths[b] - 1}
        batch_idx = np.arange(B)
        last_hidden_data = hidden.data[batch_idx, gather_indices, :][:, None, :]  # [B, 1, embed_dim]
        last_hidden = Tensor(last_hidden_data, (hidden,), "LastTokenPool")

        def _pool_backward():
            if hidden.requires_grad and last_hidden.grad is not None:
                if hidden.grad is None:
                    hidden.grad = np.zeros_like(hidden.data, dtype=np.float64)
                hidden.grad[batch_idx, gather_indices, :] += last_hidden.grad[:, 0, :]

        last_hidden._backward = _pool_backward

        # Project to scalar reward: [B, 1, 1] -> squeeze to [B, 1]
        reward = self.reward_head(last_hidden)
        reward_squeezed_data = reward.data.squeeze(axis=-1)  # [B, 1]
        out = Tensor(reward_squeezed_data, (reward,), "RewardSqueeze")

        def _squeeze_backward():
            if reward.requires_grad and out.grad is not None:
                _accumulate_grad(reward, out.grad[:, :, None])

        out._backward = _squeeze_backward
        return out


def bradley_terry_reward_loss(r_chosen: Tensor, r_rejected: Tensor, margin: float = 0.0) -> Tensor:
    """Computes pairwise Bradley-Terry preference loss: L = -log(sigmoid(r_chosen - r_rejected - margin))."""
    if r_chosen.shape != r_rejected.shape:
        raise ValueError(
            f"Shape mismatch in bradley_terry_reward_loss: r_chosen shape {r_chosen.shape} "
            f"!= r_rejected shape {r_rejected.shape}"
        )
    diff_data = (r_chosen.data - r_rejected.data) - margin
    # Numerically stable log-sigmoid: -log(1 / (1 + exp(-x))) = log(1 + exp(-x))
    # Or for x >= 0: -log(1 / (1 + exp(-x))) = log(1 + exp(-x))
    # softplus(-x)
    loss_data = np.log1p(np.exp(-np.clip(diff_data, -50.0, 50.0)))
    mean_loss = float(np.mean(loss_data))

    out = Tensor(mean_loss, (r_chosen, r_rejected), "BradleyTerryLoss")

    def _backward():
        # d/d(r_chosen) = - (1 - sigmoid(diff)) / N = - sigmoid(-diff) / N
        N = r_chosen.data.shape[0]
        sig_neg_diff = 1.0 / (1.0 + np.exp(np.clip(diff_data, -50.0, 50.0)))
        grad_scale = float(out.grad if out.grad is not None else 1.0) / N

        d_chosen = -sig_neg_diff * grad_scale
        d_rejected = sig_neg_diff * grad_scale

        if r_chosen.requires_grad:
            _accumulate_grad(r_chosen, d_chosen)
        if r_rejected.requires_grad:
            _accumulate_grad(r_rejected, d_rejected)

    out._backward = _backward
    return out


def compute_policy_gradient_step(
    policy_model: ParallaxLM,
    tokens: np.ndarray,
    reward: float,
    baseline: float = 0.0,
    kl_coef: float = 0.0,
    ref_logits: Optional[np.ndarray] = None,
) -> float:
    """Performs a REINFORCE policy gradient update step on the policy model.

    Advantage A = reward - baseline
    Loss = - A * sum_t log P(y_t | y_<t) + kl_coef * KL(P || P_ref)
    """
    advantage = reward - baseline
    logits = policy_model(tokens)  # [1, T, V]
    logits_data = logits.data[0]   # [T, V]
    T, V = logits_data.shape

    # Target sequence is tokens shifted by 1
    target_ids = tokens[0, 1:]
    L = len(target_ids)

    # Softmax log probabilities
    max_l = np.max(logits_data[:L], axis=-1, keepdims=True)
    exp_l = np.exp(logits_data[:L] - max_l)
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    log_probs = (logits_data[:L] - max_l) - np.log(np.sum(exp_l, axis=-1, keepdims=True))

    target_log_probs = log_probs[np.arange(L), target_ids]
    policy_loss = - advantage * float(np.sum(target_log_probs))

    # Analytical gradient backprop into logits:
    # dL / dlogits_t = - advantage * (1_target - probs)
    dlogits = np.zeros_like(logits_data)
    dlogits[:L] = probs.copy()
    dlogits[np.arange(L), target_ids] -= 1.0
    dlogits[:L] *= advantage

    if kl_coef != 0.0:
        if ref_logits is None:
            raise ValueError("kl_coef is non-zero but ref_logits was not provided.")
        ref_arr = ref_logits[0] if ref_logits.ndim == 3 else ref_logits
        ref_slice = ref_arr[:L]
        max_r = np.max(ref_slice, axis=-1, keepdims=True)
        exp_r = np.exp(ref_slice - max_r)
        ref_log_probs = (ref_slice - max_r) - np.log(np.sum(exp_r, axis=-1, keepdims=True))
        log_ratio = log_probs - ref_log_probs
        kl_per_t = np.sum(probs * log_ratio, axis=-1, keepdims=True)
        policy_loss += float(kl_coef * np.sum(kl_per_t))
        dlogits[:L] += kl_coef * probs * (log_ratio - kl_per_t)

    if logits.requires_grad:
        logits.backward(grad=dlogits[None, :, :])

    return policy_loss


def compute_sequence_log_probs(
    logits: Tensor,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes per-sequence summed log-probabilities of target tokens given logits,

    along with analytical gradients with respect to logits.

    Args:
        logits: Tensor of shape [B, T, V]
        targets: np.ndarray of shape [B, T] with token indices
        mask: Optional binary mask [B, T] (1.0 for valid completion tokens, 0.0 for prompt/pad)

    Returns:
        seq_log_probs: np.ndarray of shape [B], sum of log probabilities per sequence
        dlogits: np.ndarray of shape [B, T, V], gradient of seq_log_probs w.r.t logits
                 d(sum log P)/d(logits) = 1_target - softmax(logits)
    """
    logits_data = logits.data  # [B, T, V]
    B, T, V = logits_data.shape

    # Numerically stable softmax
    max_l = np.max(logits_data, axis=-1, keepdims=True)
    exp_l = np.exp(logits_data - max_l)
    sum_exp = np.sum(exp_l, axis=-1, keepdims=True)
    probs = exp_l / sum_exp
    log_probs = (logits_data - max_l) - np.log(sum_exp)

    # Gather log probs for target indices
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    selected_log_probs = log_probs[b_idx, t_idx, targets]  # [B, T]

    if mask is not None:
        selected_log_probs = selected_log_probs * mask
        active_mask = mask[:, :, None]
    else:
        active_mask = 1.0

    seq_log_probs = np.sum(selected_log_probs, axis=-1)  # [B]

    # Analytical gradient: d(sum_t log P(y_t))/d(logits) = 1_target - probs
    dlogits = np.zeros_like(logits_data)
    dlogits[b_idx, t_idx, targets] = 1.0
    dlogits = (dlogits - probs) * active_mask

    return seq_log_probs, dlogits


def dpo_loss(
    policy_chosen_logits: Tensor,
    policy_rejected_logits: Tensor,
    ref_chosen_logits: np.ndarray,
    ref_rejected_logits: np.ndarray,
    chosen_targets: np.ndarray,
    rejected_targets: np.ndarray,
    beta: float = 0.1,
    chosen_mask: Optional[np.ndarray] = None,
    rejected_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Computes the Direct Preference Optimization (DPO) loss from first principles:

    L_DPO = - E_{(x, y_w, y_l)} [ log sigma( beta * log(pi_theta(y_w|x) / pi_ref(y_w|x))
                                           - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)) ) ]

    Implicit reward: r(x, y) = beta * (log pi_theta(y|x) - log pi_ref(y|x))

    Gradients are directly accumulated into policy_chosen_logits.grad and
    policy_rejected_logits.grad.

    Returns:
        loss_val: scalar float value of the DPO loss
        implicit_reward_chosen: float reward score of chosen candidate
        implicit_reward_rejected: float reward score of rejected candidate
    """
    # 1. Compute policy log probabilities and gradient accumulators
    pi_logp_w, d_pi_w = compute_sequence_log_probs(policy_chosen_logits, chosen_targets, chosen_mask)
    pi_logp_l, d_pi_l = compute_sequence_log_probs(policy_rejected_logits, rejected_targets, rejected_mask)

    # 2. Compute reference log probabilities (treated as constants, no gradient)
    ref_w_tensor = Tensor(ref_chosen_logits, requires_grad=False)
    ref_l_tensor = Tensor(ref_rejected_logits, requires_grad=False)
    ref_logp_w, _ = compute_sequence_log_probs(ref_w_tensor, chosen_targets, chosen_mask)
    ref_logp_l, _ = compute_sequence_log_probs(ref_l_tensor, rejected_targets, rejected_mask)

    # 3. Log ratios & implicit rewards
    log_ratio_w = pi_logp_w - ref_logp_w
    log_ratio_l = pi_logp_l - ref_logp_l

    implicit_r_w = beta * log_ratio_w
    implicit_r_l = beta * log_ratio_l

    # Margin h = beta * (log_ratio_w - log_ratio_l)
    h = implicit_r_w - implicit_r_l

    # L_DPO = - log sigma(h) = softplus(-h) = log(1 + exp(-h))
    # Numerically stable calculation:
    loss_per_sample = np.where(h > 0, np.log1p(np.exp(-h)), -h + np.log1p(np.exp(h)))
    loss_val = float(np.mean(loss_per_sample))

    # 4. Analytical Gradients w.r.t logits:
    # d L_DPO / d h = - (1 - sigma(h)) = - sigma(-h)
    # sigma(-h) = 1 / (1 + exp(h))
    sig_neg_h = 1.0 / (1.0 + np.exp(np.clip(h, -50, 50)))  # Shape: [B]
    B = float(len(h))
    grad_scale = sig_neg_h / B  # [B]

    # dL / d(logits_w) = - beta * grad_scale * d_pi_w
    # dL / d(logits_l) = + beta * grad_scale * d_pi_l
    grad_w = - beta * grad_scale[:, None, None] * d_pi_w
    grad_l = + beta * grad_scale[:, None, None] * d_pi_l

    # Execute backward pass through the DAG with the calculated output gradients
    if policy_chosen_logits.requires_grad:
        policy_chosen_logits.backward(grad=grad_w)
    if policy_rejected_logits.requires_grad:
        policy_rejected_logits.backward(grad=grad_l)

    return loss_val, float(np.mean(implicit_r_w)), float(np.mean(implicit_r_l))

