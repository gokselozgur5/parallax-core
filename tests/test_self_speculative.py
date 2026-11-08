from parallax.self_speculative import SelfSpeculativeEngine


def test_self_speculative_engine_init() -> None:
    engine = SelfSpeculativeEngine(exit_layer=2, total_layers=6)
    assert engine.exit_layer == 2
    assert engine.total_layers == 6

def test_draft_tokens_generation() -> None:
    engine = SelfSpeculativeEngine(exit_layer=2, total_layers=4)
    # Mock early-exit draft function that appends token + 1
    mock_draft = lambda seq: seq[-1] + 1
    drafts = engine.draft_tokens([10], mock_draft, num_draft=3)
    assert drafts == [11, 12, 13]

def test_verify_tokens_acceptance_and_correction() -> None:
    engine = SelfSpeculativeEngine(exit_layer=2, total_layers=4)
    prompt = [10]
    drafts = [11, 12, 13]
    # Full model accepts 11, 12, but rejects 13 and predicts 99
    # preds at indices 0, 1, 2 correspond to next token after prompt[0], prompt[0]+11, etc.
    mock_full_eval = lambda seq: [11, 12, 99, 100]
    accepted = engine.verify_tokens(prompt, drafts, mock_full_eval)
    assert accepted == [11, 12, 99]
