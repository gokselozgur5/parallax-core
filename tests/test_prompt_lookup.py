from parallax.prompt_lookup import find_ngram_candidate


def test_find_ngram_candidate_exact_match():
    # Sequence: [1, 2, 3, 4, 5, 10, 1, 2, 3] -> trailing 3-gram is [1, 2, 3]
    # Prior match at index 0-2 -> continuation is [4, 5, 10]
    tokens = [1, 2, 3, 4, 5, 10, 1, 2, 3]
    cand = find_ngram_candidate(tokens, ngram_size=3, max_candidate_len=3)
    assert cand == [4, 5, 10]


def test_find_ngram_candidate_no_match():
    tokens = [1, 2, 3, 4, 5, 6]
    cand = find_ngram_candidate(tokens, ngram_size=3, max_candidate_len=3)
    assert cand is None
