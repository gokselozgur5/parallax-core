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


from parallax.prompt_lookup import prompt_lookup_step


def test_prompt_lookup_step_acceleration():
    # Sequence oracle where token after 3 is 4, after 4 is 5, etc.
    def oracle(seq):
        return seq[-1] + 1

    # Prompt contains repetition pattern: [1, 2, 3, 4, 5, 99, 1, 2, 3]
    tokens = [1, 2, 3, 4, 5, 99, 1, 2, 3]

    accepted, next_tok = prompt_lookup_step(tokens, oracle, ngram_size=3, max_candidate_len=2)

    # Candidate extracted: [4, 5]
    # Oracle verifies: after 3 is 4 (match), after 4 is 5 (match)
    # Bonus token after 5 is 6
    assert accepted == [4, 5]
    assert next_tok == 6
