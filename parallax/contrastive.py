"""
Contrastive Search decoding (Su et al., 2022) mathematical primitives.
"""

from typing import List, Tuple
import numpy as np

from parallax.tensor import no_grad


def compute_max_cosine_similarity(
    candidate_vec: np.ndarray,
    history_matrix: np.ndarray,
) -> float:
    """Computes max cosine similarity between a candidate vector and past token states."""
    if history_matrix.shape[0] == 0:
        return 0.0

    cand_norm = np.linalg.norm(candidate_vec) + 1e-12
    hist_norms = np.linalg.norm(history_matrix, axis=-1) + 1e-12

    dots = np.dot(history_matrix, candidate_vec)
    cos_sims = dots / (hist_norms * cand_norm)
    return float(np.max(cos_sims))


def score_contrastive_candidate(
    token_prob: float,
    candidate_vec: np.ndarray,
    history_matrix: np.ndarray,
    alpha: float = 0.6,
) -> float:
    """Computes the contrastive search score: (1 - alpha) * prob - alpha * max_sim."""
    max_sim = compute_max_cosine_similarity(candidate_vec, history_matrix)
    return float((1.0 - alpha) * token_prob - alpha * max_sim)


def contrastive_search_step(
    model,
    input_ids: List[int],
    history_hidden_states: List[np.ndarray],
    k: int = 4,
    alpha: float = 0.6,
) -> Tuple[int, np.ndarray]:
    """Selects the next token using Contrastive Search with top-k candidate pruning."""
    idx = np.array([input_ids], dtype=np.int64)
    with no_grad():
        logits, hidden = model(idx, return_hidden=True)

    next_token_logits = logits.data[0, -1, :]
    max_l = np.max(next_token_logits)
    exp_l = np.exp(next_token_logits - max_l)
    probs = exp_l / np.sum(exp_l)

    # Top-K candidate set S_k
    topk_candidates = np.argsort(probs)[-k:][::-1]

    history_matrix = (
        np.stack(history_hidden_states, axis=0)
        if len(history_hidden_states) > 0
        else np.empty((0, hidden.shape[-1]))
    )

    best_score = -np.inf
    best_token = int(topk_candidates[0])
    best_candidate_hidden = hidden.data[0, -1, :]

    for cand in topk_candidates:
        cand_seq = input_ids + [int(cand)]
        with no_grad():
            _, cand_hidden = model(np.array([cand_seq], dtype=np.int64), return_hidden=True)
        h_cand = cand_hidden.data[0, -1, :]

        score = score_contrastive_candidate(
            token_prob=probs[cand],
            candidate_vec=h_cand,
            history_matrix=history_matrix,
            alpha=alpha,
        )
        if score > best_score:
            best_score = score
            best_token = int(cand)
            best_candidate_hidden = h_cand

    return best_token, best_candidate_hidden


def generate_contrastive(
    model,
    prompt_tokens: List[int],
    max_new_tokens: int = 15,
    k: int = 4,
    alpha: float = 0.6,
) -> List[int]:
    """Autoregressively generates text using Contrastive Search."""
    tokens = list(prompt_tokens)
    idx = np.array([tokens], dtype=np.int64)
    with no_grad():
        _, hidden = model(idx, return_hidden=True)
    history_hidden_states: List[np.ndarray] = [hidden.data[0, i, :] for i in range(len(tokens))]

    for _ in range(max_new_tokens):
        if len(tokens) >= model.config.max_seq_len:
            break
        next_tok, next_h = contrastive_search_step(
            model,
            tokens,
            history_hidden_states,
            k=k,
            alpha=alpha,
        )
        tokens.append(next_tok)
        history_hidden_states.append(next_h)

    return tokens



