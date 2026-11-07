# Parallax Core — Roadmap & Atomic Work Breakdown

## Mission
Build a zero-dependency, mathematically verified Autograd and Decoder-Only Transformer architecture from scratch in pure Python and NumPy. Every single component must be accompanied by mathematical proofs, shape tracking, and automated gradient tests.

---

## Phase 1: Computation Graph & Scalar Autograd Engine (M1)

### [Issue #1] Core Scalar `Value` Primitives & Forward Arithmetic
- **Scope:** Define `Value(data: float, _children=(), _op='')`.
- **Operations:** `__add__`, `__mul__`, `__neg__`, `__sub__`, `__truediv__`, `__pow__`.
- **Deliverables:** `parallax/engine.py`, `tests/test_scalar_forward.py`.
- **Target Line Delta:** ~50 lines.

### [Issue #2] Topological DAG Traversal & Backward Chain Rule
- **Scope:** Implement reverse topological sort over DAG nodes and execute local `_backward` closures.
- **Deliverables:** `Value.backward()`, `tests/test_scalar_backward.py`.
- **Validation:** Compare against analytical calculus derivatives for composite scalar functions.

### [Issue #3] Automated Finite-Difference Numerical Gradient Checker
- **Scope:** Implement `numerical_gradient(fn, *inputs, eps=1e-5)` utility to automate gradient cross-checks.
- **Deliverables:** `parallax/gradcheck.py`, `tests/test_gradcheck.py`.
- **Verification Rule:** Relative error $< 10^{-5}$.

---

## Phase 2: Non-Linearities & Scalar Activation Suite (M2)

### [Issue #4] ReLU & GeLU Activations with Backward Closures
- **Scope:** Piecewise linear ReLU and Gaussian Error Linear Unit (GeLU) approximation.
- **Deliverables:** `Value.relu()`, `Value.gelu()`, `tests/test_activations.py`.

### [Issue #5] Transcendental Functions: Exp, Log, Tanh, Sigmoid
- **Scope:** Numerically stable transcendental operations with analytical derivative closures.
- **Deliverables:** `Value.exp()`, `Value.log()`, `Value.tanh()`, `Value.sigmoid()`.

---

## Phase 3: Tensor Engine & Broadcasting (M3)

### [Issue #6] N-Dimensional `Tensor` Class with Strided Memory & Metadata
- **Scope:** Multi-dimensional tensor wrapper over raw NumPy arrays maintaining DAG graph connections.
- **Deliverables:** `parallax/tensor.py`, `tests/test_tensor_shapes.py`.

### [Issue #7] Broadcast-Aware Tensor Arithmetic & Matrix Multiplication (`matmul`)
- **Scope:** Backward pass handling axis expansion, broadcasting sum-reductions, and 2D/3D batch matrix multiplication.
- **Deliverables:** `Tensor.matmul()`, `Tensor.transpose()`, `tests/test_matmul_backward.py`.

---

## Phase 4: Neural Network Abstractions & Normalization (M4)

### [Issue #8] Modular Base: `Module`, `Parameter`, and Linear Projection Layer
- **Scope:** PyTorch-style base class with `parameters()` generator and zero-grad routines.
- **Deliverables:** `parallax/nn.py` (`Module`, `Parameter`, `Linear`), `tests/test_linear.py`.

### [Issue #9] Normalization Primitives: LayerNorm & RMSNorm
- **Scope:** Channel-wise standardization, learnable affine scales/biases, backward variance propagation.
- **Deliverables:** `LayerNorm`, `RMSNorm`, `tests/test_norm.py`.

### [Issue #10] Regularization: Dropout & Residual Connections
- **Scope:** Inverted dropout with scaling, residual add-norm identity stream stability.
- **Deliverables:** `Dropout`, `Residual`, `tests/test_residual.py`.

---

## Phase 5: Self-Attention & Positional Dynamics (M5)

### [Issue #11] Scaled Dot-Product Attention & Causal Lower-Triangular Mask
- **Scope:** Compute $\text{Softmax}(QK^T / \sqrt{d_k}) V$ with autoregressive masking ($-\infty$ upper triangle).
- **Deliverables:** `parallax/attention.py` (`ScaledDotProductAttention`), `tests/test_causal_mask.py`.

### [Issue #12] Multi-Head Attention (MHA) Projection & Splitting
- **Scope:** Linear projections for Query, Key, Value; head partitioning; causal execution; output projection.
- **Deliverables:** `MultiHeadAttention`, `tests/test_mha.py`.

### [Issue #13] Rotary Positional Embedding (RoPE)
- **Scope:** Complex-valued / 2D rotation of query-key pairs to preserve relative distance.
- **Deliverables:** `RotaryEmbedding`, `tests/test_rope.py`.

---

## Phase 6: Decoder-Only Transformer Architecture (M6)

### [Issue #14] Transformer Block Integration (Pre-LN Architecture)
- **Scope:** Stacking RMSNorm/LayerNorm -> MHA -> Residual -> SwiGLU FFN -> Residual.
- **Deliverables:** `parallax/models/transformer_block.py`, `tests/test_transformer_block.py`.

### [Issue #15] Full Decoder-Only Model & Unembedding Head
- **Scope:** Token embedding lookup table, N-layer transformer stack, final norm, linear logit projection.
- **Deliverables:** `parallax/models/gpt.py`, `tests/test_gpt_forward.py`.

---

## Phase 7: Training Loop, Cross-Entropy & Generation (M7)

### [Issue #16] Numerically Stable Categorical Cross-Entropy Loss
- **Scope:** Log-sum-exp trick to prevent underflow/overflow; softmax gradient backpropagation.
- **Deliverables:** `parallax/loss.py`, `tests/test_cross_entropy.py`.

### [Issue #17] AdamW Optimizer from Scratch
- **Scope:** First and second momentum tracking, bias correction, decoupled weight decay.
- **Deliverables:** `parallax/optim.py` (`AdamW`), `tests/test_adamw.py`.

### [Issue #18] Character / Byte-Pair Tokenizer & Autoregressive Generator
- **Scope:** Encoding, decoding, greedy sampling, temperature scaling, top-k / top-p (nucleus) filtering.
- **Deliverables:** `parallax/tokenizer.py`, `parallax/generate.py`, `tests/test_generation.py`.

---

## Phase 8: Execution Tracing, Graphviz DAG & Benchmarking (M8)

### [Issue #19] Visual DAG Exporter (Graphviz & Mermaid)
- **Scope:** Export computation graphs as DOT and Mermaid format for visual inspection.
- **Deliverables:** `parallax/viz.py`, `examples/export_graph.py`.

### [Issue #20] Micro-Benchmark Suite & Gradient Sanity Dashboard
- **Scope:** Throughput (tokens/sec), peak memory consumption, and end-to-end gradient proof documentation.
- **Deliverables:** `benchmarks/run_benchmarks.py`, `docs/benchmark_report.md`.
