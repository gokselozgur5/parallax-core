"""
Comprehensive verification test suite for Batch 6 viz, bench, soft_capping, and flash_attn fixes.
Covers:
- Issue #321: bench inference throughput using actual generated tokens count
- Issue #322: Graphviz DOT label escaping for quotes, <, >, {, }, |
- Issue #323: Mermaid diagram label escaping using #quot; entity
- Issue #325: Soft-capping finite-difference gradient check
- Issue #326: Reference attention causality invariance under future key/value perturbation
- Issue #327: Flash attention backward gradcheck across dq, dk, dv on all/sampled entries (including non-causal)
"""

import numpy as np

from parallax.bench import benchmark_inference
from parallax.engine import Value
from parallax.flash_attn import (
    flash_attention_backward,
    flash_attention_forward,
    reference_attention,
)
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.soft_capping import LogitSoftCapper
from parallax.tensor import Tensor
from parallax.viz import to_dot, to_mermaid


def test_bench_inference_throughput_uses_actual_generated_tokens():
    """Verify benchmark_inference calculates throughput using actual tokens generated."""
    config = ParallaxLMConfig(
        vocab_size=32,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        use_rope=True,
    )
    model = ParallaxLM(config)

    prompt_len = 4
    gen_len = 6
    uncached, cached = benchmark_inference(model, prompt_len=prompt_len, gen_len=gen_len, num_runs=2)

    assert uncached.total_tokens == gen_len
    assert cached.total_tokens == gen_len
    assert uncached.tokens_per_second > 0
    assert cached.tokens_per_second > 0
    # Expected tok/s = total_tokens / mean_s
    expected_uncached_tok_sec = uncached.total_tokens / (uncached.latency_ms / 1000.0)
    expected_cached_tok_sec = cached.total_tokens / (cached.latency_ms / 1000.0)
    assert np.isclose(uncached.tokens_per_second, expected_uncached_tok_sec)
    assert np.isclose(cached.tokens_per_second, expected_cached_tok_sec)


def test_to_dot_escapes_special_characters():
    """Verify Graphviz DOT export escapes quotes, <, >, {, }, and | characters."""
    v1 = Value(2.0, label='say "hi"')
    v2 = Value(3.0, label='x<y & z>w')
    v3 = Value(4.0, label='50% of {a|b}')
    res = v1 + v2 + v3

    dot_str = to_dot(res)
    assert 'say \\"hi\\"' in dot_str
    assert 'x\\<y' in dot_str
    assert 'z\\>w' in dot_str
    assert '50% of \\{a\\|b\\}' in dot_str
    assert 'digraph G {' in dot_str
    assert dot_str.endswith('}')


def test_to_dot_tensor_escapes_special_characters():
    """Verify DOT export for Tensor nodes also escapes special characters."""
    t = Tensor(np.array([[1.0, 2.0]]), label='tensor<with "quotes" | {braces}>')
    loss = t.sum()
    dot_str = to_dot(loss)
    assert 'tensor\\<with \\"quotes\\" \\| \\{braces\\}\\>' in dot_str


def test_to_mermaid_escapes_quotes_with_hashquot():
    """Verify Mermaid export escapes double quotes using #quot; entity."""
    v = Value(1.5, label='say "hello" now')
    res = v * 2.0
    mermaid_str = to_mermaid(res)

    assert 'say #quot;hello#quot; now' in mermaid_str
    assert '\\"' not in mermaid_str


def test_to_mermaid_tensor_escapes_quotes():
    """Verify Mermaid export for Tensor nodes escapes double quotes using #quot;."""
    t = Tensor(np.array([1.0, 2.0]), label='my "feature" tensor')
    loss = t.sum()
    mermaid_str = to_mermaid(loss)

    assert 'my #quot;feature#quot; tensor' in mermaid_str
    assert '\\"' not in mermaid_str


def test_soft_capper_finite_difference_gradient():
    """Finite difference numerical gradient test for LogitSoftCapper."""
    cap = 15.0
    capper = LogitSoftCapper(soft_cap=cap)

    np.random.seed(123)
    logits = np.random.uniform(-30.0, 30.0, size=(10,)).astype(np.float64)
    grad_out = np.random.randn(10).astype(np.float64)

    analytic = capper.backward(grad_out, logits)

    eps = 1e-6
    numerical = np.zeros_like(logits)
    for i in range(len(logits)):
        lp = logits.copy()
        lp[i] += eps
        lm = logits.copy()
        lm[i] -= eps
        out_p = capper.forward(lp)
        out_m = capper.forward(lm)
        numerical[i] = np.sum((out_p - out_m) * grad_out) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numerical, atol=1e-5, rtol=1e-4)


def test_reference_attention_future_perturbation_invariance():
    """Verify strict causal mask invariance: mutating future tokens does not affect past outputs."""
    np.random.seed(777)
    B, H, T, d = 2, 2, 6, 8
    q = np.random.randn(B, H, T, d)
    k = np.random.randn(B, H, T, d)
    v = np.random.randn(B, H, T, d)

    base_out = reference_attention(q, k, v, is_causal=True)

    # For each position pivot t, perturb tokens > t in k and v
    for pivot in [1, 2, 3, 4]:
        k_perturbed = k.copy()
        v_perturbed = v.copy()
        k_perturbed[:, :, pivot:, :] += np.random.randn(B, H, T - pivot, d) * 3.0
        v_perturbed[:, :, pivot:, :] += np.random.randn(B, H, T - pivot, d) * 3.0

        pert_out = reference_attention(q, k_perturbed, v_perturbed, is_causal=True)
        # Outputs at positions <= pivot - 1 must be strictly invariant
        np.testing.assert_allclose(base_out[:, :, :pivot, :], pert_out[:, :, :pivot, :], atol=1e-12)


def test_flash_attention_backward_all_gradcheck_non_causal_and_asymmetric():
    """Gradcheck dq, dk, dv across full entries on non-causal attention with asymmetric blocks."""
    np.random.seed(42)
    B, H, T, d = 1, 1, 5, 4
    q = np.random.randn(B, H, T, d)
    k = np.random.randn(B, H, T, d)
    v = np.random.randn(B, H, T, d)
    do = np.random.randn(B, H, T, d)

    # block size not dividing T (T=5, block_r=3, block_c=2)
    o, lse = flash_attention_forward(q, k, v, is_causal=False, block_r=3, block_c=2)
    dq, dk, dv = flash_attention_backward(
        do, o, q, k, v, lse, is_causal=False, block_r=3, block_c=2
    )

    eps = 1e-5
    # Check all entries of dq, dk, dv
    for t in range(T):
        for c in range(d):
            # dq
            qp, qm = q.copy(), q.copy()
            qp[0, 0, t, c] += eps
            qm[0, 0, t, c] -= eps
            op, _ = flash_attention_forward(qp, k, v, is_causal=False, block_r=3, block_c=2)
            om, _ = flash_attention_forward(qm, k, v, is_causal=False, block_r=3, block_c=2)
            num_dq = np.sum((op - om) * do) / (2 * eps)
            np.testing.assert_allclose(dq[0, 0, t, c], num_dq, atol=1e-5, rtol=1e-4)

            # dk
            kp, km = k.copy(), k.copy()
            kp[0, 0, t, c] += eps
            km[0, 0, t, c] -= eps
            op_k, _ = flash_attention_forward(q, kp, v, is_causal=False, block_r=3, block_c=2)
            om_k, _ = flash_attention_forward(q, km, v, is_causal=False, block_r=3, block_c=2)
            num_dk = np.sum((op_k - om_k) * do) / (2 * eps)
            np.testing.assert_allclose(dk[0, 0, t, c], num_dk, atol=1e-5, rtol=1e-4)

            # dv
            vp, vm = v.copy(), v.copy()
            vp[0, 0, t, c] += eps
            vm[0, 0, t, c] -= eps
            op_v, _ = flash_attention_forward(q, k, vp, is_causal=False, block_r=3, block_c=2)
            om_v, _ = flash_attention_forward(q, k, vm, is_causal=False, block_r=3, block_c=2)
            num_dv = np.sum((op_v - om_v) * do) / (2 * eps)
            np.testing.assert_allclose(dv[0, 0, t, c], num_dv, atol=1e-5, rtol=1e-4)
