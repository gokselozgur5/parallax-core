"""
Unit tests for computation graph export and attention heatmaps (M8).
"""

import numpy as np

from parallax.engine import Value
from parallax.tensor import Tensor
from parallax.attention import CausalSelfAttention
from parallax.viz import (
    trace_graph,
    to_mermaid,
    to_dot,
    format_attention_heatmap,
)


def test_scalar_graph_tracing_and_mermaid():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = a + b
    c.label = "c"
    d = c * 4.0
    d.label = "d"
    d.backward()

    # a, b, c, 4.0 (scalar literal wrapped as Value), d -> 5 nodes total
    nodes, edges = trace_graph(d)
    assert len(nodes) == 5
    assert len(edges) == 4

    mermaid_str = to_mermaid(d)
    assert "graph LR" in mermaid_str
    assert "a: data: 2.0000" in mermaid_str
    assert "d: data: -4.0000" in mermaid_str
    assert '(["*"])' in mermaid_str


def test_scalar_graph_to_dot():
    x = Value(1.5, label="x")
    y = x ** 2
    y.backward()

    dot_str = to_dot(y)
    assert "digraph G {" in dot_str
    assert "x | data: 1.5000" in dot_str
    assert "**2" in dot_str
    assert dot_str.endswith("}")


def test_tensor_graph_tracing_and_mermaid():
    W = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), label="W")
    x = Tensor(np.array([[0.5], [-0.5]]), label="x")
    y = W @ x
    loss = y.sum()
    loss.backward()

    mermaid_str = to_mermaid(loss)
    assert "graph LR" in mermaid_str
    assert "W: shape: (2, 2)" in mermaid_str
    assert "matmul" in mermaid_str or "@" in mermaid_str or "sum" in mermaid_str

    dot_str = to_dot(loss)
    assert "digraph G {" in dot_str
    assert "shape: (2, 2)" in dot_str


def test_format_attention_heatmap():
    tokens = ["H", "E", "L", "L", "O"]
    T = len(tokens)
    # Generate causal attention matrix
    raw = np.random.uniform(0.1, 1.0, size=(T, T))
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    raw[mask] = 0.0
    attn_matrix = raw / np.sum(raw, axis=-1, keepdims=True)

    heatmap = format_attention_heatmap(attn_matrix, tokens=tokens)
    assert len(heatmap.splitlines()) >= T + 2
    assert "H" in heatmap
    assert "O" in heatmap


def test_causal_self_attention_records_weights():
    embed_dim = 16
    num_heads = 2
    seq_len = 4
    mha = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = Tensor(np.random.randn(1, seq_len, embed_dim))
    _ = mha(x)

    assert hasattr(mha, "last_attn_weights")
    assert mha.last_attn_weights is not None
    assert mha.last_attn_weights.shape == (1, num_heads, seq_len, seq_len)
