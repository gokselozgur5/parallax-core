"""
Mixture-of-Experts (MoE: Shazeer et al., 2017; Fedus et al., 2022) from first principles.

Implements Sparse Top-K Gating with auxiliary load balancing loss:
    Gate(x) = Softmax(TopK(x @ W_gate^T, k))
    MoE(x) = sum_{i in TopK} Gate(x)_i * Expert_i(x)

Auxiliary Load Balancing Loss:
    L_aux = alpha * num_experts * sum_{i=1}^E (f_i * P_i)
where f_i is the fraction of tokens routed to expert i, and P_i is the average gate probability.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from parallax.models.gpt import MLP, SwiGLU
from parallax.nn import Linear, Module, Parameter
from parallax.tensor import Tensor, _accumulate_grad, unbroadcast_grad


class SparseMoE(Module):
    """Sparse Top-K Mixture of Experts module replacing a single feed-forward layer.

    Args:
        embed_dim: Token representation dimension.
        num_experts: Total number of available expert networks (E).
        top_k: Number of experts activated per token (K, e.g. 2).
        use_swiglu: Whether experts use SwiGLU or standard GELU MLPs.
        aux_loss_coef: Coefficient for load balancing auxiliary loss.
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        use_swiglu: bool = False,
        aux_loss_coef: float = 0.01,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coef = aux_loss_coef

        # Router / Gating linear projection: [embed_dim -> num_experts]
        self.gate = Linear(embed_dim, num_experts, bias=False)
        self.router = self.gate

        # Expert networks
        self.experts: List[Module] = []
        for i in range(num_experts):
            expert = SwiGLU(embed_dim, hidden_dim=hidden_dim) if use_swiglu else MLP(embed_dim, hidden_dim=hidden_dim)
            self.experts.append(expert)

        # Buffer for latest auxiliary loss and routing metrics
        self.last_aux_loss: float = 0.0
        self.aux_loss: Optional[Tensor] = None
        self.last_expert_counts: np.ndarray = np.zeros(num_experts, dtype=np.int64)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with Top-K sparse routing and dynamic dispatch.

        Args:
            x: Tensor of shape [B, T, embed_dim]

        Returns:
            out: Combined expert output of shape [B, T, embed_dim]
        """
        orig_shape = x.shape
        B, T, D = orig_shape
        N = B * T  # Total number of tokens in the batch

        # 1. Gating logits: [B, T, num_experts] -> flatten to [N, num_experts]
        gate_logits = self.gate(x)
        logits_data = gate_logits.data.reshape(N, self.num_experts)

        # Softmax probabilities over all experts
        max_logits = np.max(logits_data, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_data - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # [N, E]

        # 2. Top-K Selection per token
        topk_indices = np.argsort(probs, axis=-1)[:, -self.top_k:]  # [N, K]
        # Gather top-k probabilities
        row_indices = np.arange(N)[:, None]
        topk_probs_raw = probs[row_indices, topk_indices]  # [N, K]

        # Renormalize top-k weights so they sum to 1.0 per token (when top_k > 1)
        # When top_k == 1, do not renormalize to 1.0 (which zeroes router gradient)
        if self.top_k == 1:
            topk_weights = topk_probs_raw  # [N, 1] raw softmax probability
        else:
            topk_weights = topk_probs_raw / np.sum(topk_probs_raw, axis=-1, keepdims=True)  # [N, K]

        # 3. Compute Switch Transformer style Load Balancing Auxiliary Loss
        # f_i: fraction of tokens routed to expert i
        # P_i: average probability allocated to expert i
        expert_counts = np.zeros(self.num_experts, dtype=np.float64)
        for e_idx in topk_indices.ravel():
            expert_counts[e_idx] += 1.0
        f = expert_counts / (N * self.top_k)  # [E]
        P = np.mean(probs, axis=0)           # [E]
        aux_loss = self.aux_loss_coef * float(self.num_experts * np.sum(f * P))
        self.last_aux_loss = aux_loss
        self.last_expert_counts = expert_counts.astype(np.int64)

        # Expose aux_loss as Tensor connected to gate_logits autograd graph
        aux_loss_tensor = Tensor(np.array(aux_loss, dtype=np.float64), (gate_logits,), "aux_loss")

        def _aux_backward() -> None:
            if aux_loss_tensor.grad is None or not gate_logits.requires_grad:
                return
            g = float(aux_loss_tensor.grad)
            dL_dp = (g * self.aux_loss_coef * self.num_experts / N) * np.broadcast_to(f, (N, self.num_experts))
            sum_p_dLdp = np.sum(probs * dL_dp, axis=-1, keepdims=True)
            dL_dlogits = probs * (dL_dp - sum_p_dLdp)
            _accumulate_grad(gate_logits, dL_dlogits.reshape(B, T, self.num_experts))

        aux_loss_tensor._backward = _aux_backward
        self.aux_loss = aux_loss_tensor

        # 4. Expert Execution & Accumulation
        x_flat = x.data.reshape(N, D)
        out_flat = np.zeros((N, D), dtype=np.float64)

        expert_token_indices: Dict[int, np.ndarray] = {}
        expert_out_tensors: Dict[int, Tensor] = {}
        expert_in_tensors: Dict[int, Tensor] = {}

        for e in range(self.num_experts):
            # Find tokens routed to expert e
            mask = np.any(topk_indices == e, axis=-1)
            token_idx = np.where(mask)[0]
            if len(token_idx) == 0:
                continue

            match_k = np.where(topk_indices[token_idx] == e)[1]
            e_weights = topk_weights[token_idx, match_k][:, None]  # [n_tokens, 1]

            tokens_for_e = x_flat[token_idx]
            tensor_for_e = Tensor(tokens_for_e, requires_grad=True)
            expert_out = self.experts[e](tensor_for_e)

            out_flat[token_idx] += e_weights * expert_out.data

            expert_token_indices[e] = token_idx
            expert_out_tensors[e] = expert_out
            expert_in_tensors[e] = tensor_for_e

        out = Tensor(out_flat.reshape(B, T, D), (x, gate_logits), "SparseMoE")

        def _backward() -> None:
            if out.grad is None:
                return

            grad_flat = out.grad.reshape(N, D)
            dx_flat = np.zeros((N, D), dtype=np.float64)
            dlogits_flat = np.zeros((N, self.num_experts), dtype=np.float64)
            d_topk_weights = np.zeros((N, self.top_k), dtype=np.float64)

            # Backprop through active experts
            for e in range(self.num_experts):
                if e not in expert_token_indices:
                    continue

                token_idx = expert_token_indices[e]
                match_k = np.where(topk_indices[token_idx] == e)[1]
                e_weights = topk_weights[token_idx, match_k][:, None]

                # dL / d(expert_out) = grad * e_weights
                expert_out_grad = grad_flat[token_idx] * e_weights
                expert_out = expert_out_tensors[e]
                expert_out.backward(grad=expert_out_grad)

                # Gradient flowing back into input x from this expert
                tin = expert_in_tensors[e]
                if tin.grad is not None:
                    dx_flat[token_idx] += tin.grad

                # Routing gradient into gate logits
                d_weight = np.sum(grad_flat[token_idx] * expert_out.data, axis=-1)
                d_topk_weights[token_idx, match_k] = d_weight

            # Propagate through exact Top-K Softmax Jacobian
            if self.top_k == 1:
                dL_dprobs = np.zeros_like(probs)
                dL_dprobs[row_indices, topk_indices] = d_topk_weights
                sum_p_dLdp = np.sum(probs * dL_dprobs, axis=-1, keepdims=True)
                dlogits_flat = probs * (dL_dprobs - sum_p_dLdp)
            else:
                sum_dw = np.sum(d_topk_weights * topk_weights, axis=-1, keepdims=True)
                d_topk_logits = topk_weights * (d_topk_weights - sum_dw)
                dlogits_flat[row_indices, topk_indices] = d_topk_logits

            # Propagate to x and gate_logits
            if x.requires_grad:
                _accumulate_grad(x, dx_flat.reshape(B, T, D))

            if gate_logits.requires_grad:
                dgate = dlogits_flat.reshape(B, T, self.num_experts)
                _accumulate_grad(gate_logits, dgate)

        out._backward = _backward
        return out
