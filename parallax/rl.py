"""
First-principles Reward Modeling and Reinforcement Learning (RLHF / Policy Optimization).

Implements:
1. RewardModel: Language model backbone with a scalar head predicting alignment/human-preference scores
2. Bradley-Terry Preference Loss: Loss = -log(sigmoid(r_chosen - r_rejected))
3. REINFORCE / Policy Gradient Alignment Loop: Policy updates with reward feedback and KL regularization against reference model
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor, unbroadcast_grad


class RewardModel(Module):
    """Scalar Reward Model that outputs a scalar score r for any text sequence."""

    def __init__(self, config: ParallaxLMConfig):
        super().__init__()
        self.config = config
        self.backbone = ParallaxLM(config)
        # Linear projection from hidden dimension to single scalar reward
        self.reward_head = Linear(config.embed_dim, 1, bias=False)

    def forward(self, idx: np.ndarray) -> Tensor:
        # idx: [batch_size, seq_len]
        logits, hidden = self.backbone(idx, return_hidden=True)
        # Shape: [batch_size, seq_len, embed_dim]
        # Pool/select the last token representation h_T
        last_hidden_data = hidden.data[:, -1:, :]  # [B, 1, embed_dim]
        last_hidden = Tensor(last_hidden_data, (hidden,), "LastTokenPool")

        def _pool_backward():
            if hidden.requires_grad and hidden.grad is not None and last_hidden.grad is not None:
                hidden.grad[:, -1:, :] += last_hidden.grad

        last_hidden._backward = _pool_backward

        # Project to scalar reward: [B, 1, 1] -> squeeze to [B, 1]
        reward = self.reward_head(last_hidden)
        reward_squeezed_data = reward.data.squeeze(axis=-1)  # [B, 1]
        out = Tensor(reward_squeezed_data, (reward,), "RewardSqueeze")

        def _squeeze_backward():
            if reward.requires_grad and reward.grad is not None and out.grad is not None:
                reward.grad += out.grad[:, :, None]

        out._backward = _squeeze_backward
        return out


def bradley_terry_reward_loss(r_chosen: Tensor, r_rejected: Tensor, margin: float = 0.0) -> Tensor:
    """Computes pairwise Bradley-Terry preference loss: L = -log(sigmoid(r_chosen - r_rejected - margin))."""
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

        if r_chosen.requires_grad and r_chosen.grad is not None:
            r_chosen.grad += unbroadcast_grad(d_chosen, r_chosen.shape)
        if r_rejected.requires_grad and r_rejected.grad is not None:
            r_rejected.grad += unbroadcast_grad(d_rejected, r_rejected.shape)

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
    input_ids = tokens[0, :-1]
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

    if logits.requires_grad and logits.grad is not None:
        logits.grad += dlogits[None, :, :]

    return policy_loss
