"""
Computation Graph and Execution Tracing Visualizer.

Provides exporters for Scalar Value and Tensor DAGs to Mermaid and Graphviz DOT,
as well as ASCII heatmaps and visualizers for Attention matrices.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from parallax.engine import Value
from parallax.tensor import Tensor


def trace_graph(root: Union[Value, Tensor]) -> Tuple[Set[Union[Value, Tensor]], Set[Tuple[Union[Value, Tensor], Union[Value, Tensor]]]]:
    """Builds a set of all nodes and edges in the computation graph rooted at root."""
    nodes: Set[Union[Value, Tensor]] = set()
    edges: Set[Tuple[Union[Value, Tensor], Union[Value, Tensor]]] = set()

    def build(v: Union[Value, Tensor]) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def to_mermaid(root: Union[Value, Tensor], direction: str = "LR") -> str:
    """Exports computation DAG to a Mermaid diagram string.

    Works cleanly with both Scalar Value and N-D Tensor nodes.
    """
    nodes, edges = trace_graph(root)
    lines: List[str] = [f"graph {direction}"]

    for n in nodes:
        uid = f"node_{id(n)}"
        if isinstance(n, Value):
            lbl = f"{n.label}: " if n.label else ""
            node_desc = f"{lbl}data: {n.data:.4f} | grad: {n.grad:.4f}"
        else:
            lbl = f"{n.label}: " if n.label else ""
            grad_status = "grad: None" if n.grad is None else f"grad: norm {np.linalg.norm(n.grad):.4f}"
            node_desc = f"{lbl}shape: {n.shape} | {grad_status}"

        # Clean string for Mermaid
        node_desc = node_desc.replace('"', '\\"')
        lines.append(f'    {uid}["{node_desc}"]')

        if n._op:
            op_uid = f"op_{id(n)}"
            op_label = n._op.replace('"', '\\"')
            lines.append(f'    {op_uid}(["{op_label}"])')
            lines.append(f"    {op_uid} --> {uid}")

    for src, dst in edges:
        src_uid = f"node_{id(src)}"
        if dst._op:
            dst_target = f"op_{id(dst)}"
        else:
            dst_target = f"node_{id(dst)}"
        lines.append(f"    {src_uid} --> {dst_target}")

    return "\n".join(lines)


def _escape_dot_label(s: str) -> str:
    """Escapes special characters in Graphviz record-based labels."""
    # Backslash first to avoid double-escaping
    s = s.replace('\\', '\\\\')
    for ch in ('"', '<', '>', '{', '}', '|'):
        s = s.replace(ch, f'\\{ch}')
    return s


def to_dot(root: Union[Value, Tensor]) -> str:
    """Exports computation DAG to a Graphviz DOT format string."""
    nodes, edges = trace_graph(root)
    dot: List[str] = [
        'digraph G {',
        '    rankdir=LR;',
        '    node [shape=record, style="filled,rounded", fontname="Courier", fontsize=10];',
        '    edge [fontname="Courier", fontsize=9];'
    ]

    for n in nodes:
        uid = f"n_{id(n)}"
        if isinstance(n, Value):
            lbl = f"{_escape_dot_label(n.label)} | " if n.label else ""
            label_str = f"{{{lbl}data: {n.data:.4f} | grad: {n.grad:.4f}}}"
        else:
            lbl = f"{_escape_dot_label(n.label)} | " if n.label else ""
            g_str = "grad: None" if n.grad is None else f"grad_norm: {np.linalg.norm(n.grad):.4e}"
            label_str = f"{{{lbl}shape: {n.shape} | {g_str}}}"

        dot.append(f'    {uid} [label="{label_str}", fillcolor="#f0f4f8"];')

        if n._op:
            op_uid = f"op_{id(n)}"
            op_label = _escape_dot_label(n._op)
            dot.append(f'    {op_uid} [shape=circle, label="{op_label}", fillcolor="#ffebee"];')
            dot.append(f"    {op_uid} -> {uid};")

    for src, dst in edges:
        src_uid = f"n_{id(src)}"
        dst_target = f"op_{id(dst)}" if dst._op else f"n_{id(dst)}"
        dot.append(f"    {src_uid} -> {dst_target};")

    dot.append("}")
    return "\n".join(dot)


def format_attention_heatmap(attn_matrix: np.ndarray, tokens: Optional[List[str]] = None) -> str:
    """Renders a 2D attention weight matrix as a high-contrast ASCII/Unicode heatmap.

    Args:
        attn_matrix: 2D array of shape [T, T] with sum along rows = 1.
        tokens: Optional list of token strings of length T.
    """
    assert attn_matrix.ndim == 2, f"Expected 2D matrix, got shape {attn_matrix.shape}"
    rows, cols = attn_matrix.shape
    if tokens is not None:
        assert len(tokens) == rows, "tokens length must match matrix dimensions"

    shades = [" ", "░", "▒", "▓", "█"]
    max_val = np.max(attn_matrix) if np.max(attn_matrix) > 0 else 1.0

    lines: List[str] = []
    
    # Header row if tokens present
    if tokens:
        header = "       " + " ".join(f"{t[:4]:>4}" for t in tokens)
        lines.append(header)
        lines.append("       " + "-" * (5 * cols))

    for r in range(rows):
        row_str = ""
        for c in range(cols):
            val = attn_matrix[r, c]
            norm_val = val / max_val
            idx = min(int(norm_val * len(shades)), len(shades) - 1)
            char = shades[idx]
            row_str += f"  {char}{char}"

        prefix = f"{tokens[r][:6]:>6} |" if tokens else f"{r:>3} |"
        lines.append(prefix + row_str)

    return "\n".join(lines)
