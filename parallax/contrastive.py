"""
Contrastive Search decoding (Su et al., 2022) mathematical primitives.
"""

from typing import List, Optional, Tuple
import numpy as np


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
