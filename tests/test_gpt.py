import numpy as np
import pytest

from parallax.grouped_gemm import GroupedGEMMDispatcher
from parallax.models.gpt import Embedding


def test_embedding_rejects_negative_token_ids():
    emb = Embedding(vocab_size=5, embed_dim=4)
    with pytest.raises((IndexError, ValueError)):
        emb(np.array([[0, -1]]))


def test_grouped_gemm_rejects_missing_or_negative_expert_ids():
    disp = GroupedGEMMDispatcher(num_experts=2)
    tokens = np.ones((4, 3))
    weights = [np.ones((3, 2)), 2.0 * np.ones((3, 2))]
    with pytest.raises((IndexError, ValueError)):
        disp.grouped_gemm_forward(tokens, [0, 1], weights)
    with pytest.raises((IndexError, ValueError)):
        disp.grouped_gemm_forward(tokens, [-1, -1, -1, -1], weights)
