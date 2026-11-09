import numpy as np
import pytest
from parallax.batching import ContinuousBatchEngine, RequestStatus, SequenceRequest
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


def test_continuous_batch_engine_multi_request():
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    engine = ContinuousBatchEngine(model=model, max_batch_size=2)

    req1 = SequenceRequest(request_id="req-1", prompt_tokens=[1, 2, 3], max_new_tokens=4)
    req2 = SequenceRequest(request_id="req-2", prompt_tokens=[4, 5], max_new_tokens=2)
    req3 = SequenceRequest(request_id="req-3", prompt_tokens=[6, 7, 8, 9], max_new_tokens=3)

    engine.add_request(req1)
    engine.add_request(req2)
    engine.add_request(req3)

    assert len(engine.waiting_queue) == 3

    # Step loop until all completed
    step_count = 0
    all_emitted = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        all_emitted.extend(step_outputs)
        step_count += 1
        assert step_count < 100, "Safety loop limit reached"

    # All requests must be completed
    assert req1.status == RequestStatus.FINISHED
    assert req2.status == RequestStatus.FINISHED
    assert req3.status == RequestStatus.FINISHED

    assert len(req1.generated_tokens) == 4
    assert len(req2.generated_tokens) == 2
    assert len(req3.generated_tokens) == 3
    assert len(engine.completed_requests) == 3
