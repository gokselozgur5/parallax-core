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


def test_continuous_batch_engine_ejects_failed_request():
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

    # Bad request with NaN temperature will trigger ValueError during sampling
    bad_req = SequenceRequest(
        request_id="bad-req",
        prompt_tokens=[1, 2],
        max_new_tokens=4,
        temperature=float("nan"),
    )
    good_req = SequenceRequest(
        request_id="good-req",
        prompt_tokens=[3, 4],
        max_new_tokens=2,
        temperature=0.0,
    )

    engine.add_request(bad_req)
    engine.add_request(good_req)

    # First step: prefill for both. bad_req will fail on sampling and be ejected.
    step_outputs = engine.step()

    assert bad_req.status == RequestStatus.FAILED
    assert bad_req.error is not None
    assert bad_req.request_id in engine.completed_requests
    assert bad_req not in engine.running_requests

    # good_req must have produced output and still be healthy
    assert any(r_id == "good-req" for r_id, _ in step_outputs)

    # Engine must continue running remaining steps without poisoning
    step_count = 0
    while engine.has_unfinished_requests():
        engine.step()
        step_count += 1
        assert step_count < 10

    assert good_req.status == RequestStatus.FINISHED
    assert len(good_req.generated_tokens) == 2


def test_continuous_batch_engine_step_runs_under_no_grad():
    from parallax.tensor import is_grad_enabled

    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )

    class SpyLM(ParallaxLM):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.grad_enabled_log = []

        def __call__(self, *args, **kwargs):
            self.grad_enabled_log.append(is_grad_enabled())
            return super().__call__(*args, **kwargs)

    spy_model = SpyLM(config)
    engine = ContinuousBatchEngine(model=spy_model, max_batch_size=2)
    req = SequenceRequest(request_id="r1", prompt_tokens=[1, 2], max_new_tokens=2)
    engine.add_request(req)

    # Step 1: prefill
    engine.step()
    # Step 2: decode
    engine.step()

    assert len(spy_model.grad_enabled_log) == 2
    assert all(enabled is False for enabled in spy_model.grad_enabled_log)


def test_continuous_batch_engine_concurrent_requests():
    import threading
    import time

    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    engine = ContinuousBatchEngine(model=model, max_batch_size=4)

    def producer(start_idx: int):
        for i in range(3):
            req = SequenceRequest(
                request_id=f"thread-{start_idx}-{i}",
                prompt_tokens=[1, 2, 3],
                max_new_tokens=2,
            )
            engine.add_request(req)
            time.sleep(0.005)

    threads = [
        threading.Thread(target=producer, args=(1,)),
        threading.Thread(target=producer, args=(2,)),
    ]
    for t in threads:
        t.start()

    # Consumer loop running concurrently
    completed = 0
    safety = 0
    while completed < 6 and safety < 100:
        outputs = engine.step()
        if not outputs:
            time.sleep(0.005)
        completed = len(engine.completed_requests)
        safety += 1

    for t in threads:
        t.join()

    # Finish any remaining
    while engine.has_unfinished_requests() and safety < 150:
        engine.step()
        safety += 1

    assert len(engine.completed_requests) == 6
    for req in engine.completed_requests.values():
        assert req.status == RequestStatus.FINISHED
        assert len(req.generated_tokens) == 2


def test_pop_completed_request_clears_kv_cache():
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

    req = SequenceRequest(request_id="pop-test", prompt_tokens=[1, 2], max_new_tokens=2)
    engine.add_request(req)

    # Step until finished
    while engine.has_unfinished_requests():
        engine.step()

    assert req.status == RequestStatus.FINISHED
    assert req.kv_cache is not None
    assert "pop-test" in engine.completed_requests

    popped = engine.pop_completed_request("pop-test", clear_kv=True)
    assert popped is req
    assert popped.kv_cache is None
    assert "pop-test" not in engine.completed_requests



