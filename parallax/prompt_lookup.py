"""Prompt Lookup Decoding (PLD) for speculative decoding without draft models.

Reference: Saxena, 2023 - Prompt Lookup Decoding.
Extracts speculative token candidates by finding matching n-gram occurrences
within the input prompt and previously generated text.
"""

from typing import List, Optional, Tuple
import numpy as np


def find_ngram_candidate(
    tokens: List[int],
    ngram_size: int = 3,
    max_candidate_len: int = 5,
) -> Optional[List[int]]:
    """Search for the trailing n-gram in historical tokens and extract candidate continuation.

    Args:
        tokens: Current full sequence of tokens.
        ngram_size: Length of suffix pattern to search for.
        max_candidate_len: Maximum number of candidate tokens to extract.

    Returns:
        List of speculative candidate tokens if an n-gram match is found, else None.
    """
    total_len = len(tokens)
    if total_len < ngram_size + 1:
        return None

    query_ngram = tokens[-ngram_size:]
    # Search backwards from total_len - ngram_size - 1 down to 0
    for i in range(total_len - ngram_size - 1, -1, -1):
        if tokens[i : i + ngram_size] == query_ngram:
            # Match found! Extract up to max_candidate_len following tokens
            cand_start = i + ngram_size
            cand_end = min(cand_start + max_candidate_len, total_len - ngram_size)
            if cand_end > cand_start:
                return tokens[cand_start:cand_end]

    return None
