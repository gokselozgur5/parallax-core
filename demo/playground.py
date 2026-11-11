#!/usr/bin/env python3
"""
Parallax Core: Interactive First-Principles AI Playground.

An interactive exploration lab for understanding gradient flow,
causal attention dynamics, RoPE rotations, and autoregressive generation.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from parallax.engine import Value
from parallax.tensor import Tensor
from parallax.attention import CausalSelfAttention, apply_rope
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.loss import cross_entropy_loss
from parallax.optim import AdamW
from parallax.generator import CharTokenizer, generate
from parallax.viz import format_attention_heatmap
from parallax.bench import count_parameters, estimate_flops_per_token, benchmark_inference, format_benchmark_table


def header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title.upper()}")
    print("=" * 65)


def experiment_1_single_neuron():
    header("Experiment 1: Watch a Single Neuron Learn (w * x + b)")
    print("Target: Teach neuron to output y = 10.0 when given x = 2.0")
    print("Initial state: w = 0.5, b = 0.0 (Initial Prediction = 1.0, Error = 9.0)\n")

    w = Value(0.5, label="w")
    b = Value(0.0, label="b")
    lr = 0.05

    for step in range(1, 16):
        x = Value(2.0, label="x")
        target = 10.0

        # Forward pass: y_hat = w * x + b
        y_hat = w * x + b
        # Loss: (y_hat - target)^2
        diff = y_hat - target
        loss = diff**2

        # Backward pass
        w.grad = 0.0
        b.grad = 0.0
        loss.backward()

        # Gradient descent update
        w.data -= lr * w.grad
        b.data -= lr * b.grad

        bar_len = int(min(loss.data * 2, 30))
        bar = "#" * bar_len
        print(f"Step {step:02d} | Loss: {loss.data:8.4f} | Pred: {y_hat.data:6.2f} | w: {w.data:5.2f}, b: {b.data:5.2f} | [{bar}]")

    print("\nResult: Neuron solved the linear equation through pure gradient descent.")


def experiment_2_attention_matrix():
    header("Experiment 2: Inspect Causal Attention Matrix")
    prompt = "HELLO"
    print(f"Simulating Causal Self-Attention over prompt tokens: {list(prompt)}\n")

    seq_len = len(prompt)
    embed_dim = 16
    num_heads = 2
    head_dim = embed_dim // num_heads

    np.random.seed(42)
    mha = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, use_rope=False)
    x = Tensor(np.random.randn(1, seq_len, embed_dim))

    # Perform forward pass to extract internal attention weights
    B, T, C = x.shape
    q = mha.q_proj(x)
    k = mha.k_proj(x)
    q_heads = np.transpose(q.data.reshape(B, T, num_heads, head_dim), (0, 2, 1, 3))
    k_heads = np.transpose(k.data.reshape(B, T, num_heads, head_dim), (0, 2, 1, 3))

    scores = (q_heads @ np.swapaxes(k_heads, -1, -2)) * mha.scale
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask[None, None, :, :], -1e9, scores)

    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    head0 = attn_weights[0, 0]  # [T, T]

    print("Tokens: " + "   ".join([f"'{ch}'" for ch in prompt]))
    print("---------------------------------------")
    for i in range(T):
        row_str = f"'{prompt[i]}' -> | "
        for j in range(T):
            val = head0[i, j]
            if val < 1e-4:
                row_str += "  0.00 "
            else:
                row_str += f" {val:5.2f} "
        print(row_str + "|")

    print("\nVisual Heatmap Representation:")
    print(format_attention_heatmap(head0, tokens=list(prompt)))

    print("\nObservation:")
    print("Notice the lower-triangular pattern: token i can ONLY attend to tokens <= i.")
    print("Future tokens are completely zeroed out (strictly non-leaking).")


def experiment_3_rope_rotation():
    header("Experiment 3: Observe Rotary Positional Embeddings (RoPE)")
    print("Rotating 2D feature coordinates across positions t=0, 1, 2, 3, 4...\n")

    # Single 2D vector [1.0, 0.0] at 5 consecutive positions
    vecs = np.zeros((1, 5, 2))
    for t in range(5):
        vecs[0, t, :] = [1.0, 0.0]

    rotated = apply_rope(vecs, seq_dim=1)[0]  # Shape: [5, 2]

    for t in range(5):
        rx, ry = rotated[t, 0], rotated[t, 1]
        angle_deg = np.degrees(np.arctan2(ry, rx))
        norm = np.linalg.norm(rotated[t])
        print(f"Pos {t} | Vector: [{rx:6.3f}, {ry:6.3f}] | Angle: {angle_deg:6.1f}° | Length: {norm:.2f}")

    print("\nObservation: Vector length remains 1.00 (invariant), but orientation encodes position.")


def experiment_4_train_micro_lm():
    header("Experiment 4: Train Micro ParallaxLM from Scratch (In-Memory)")
    corpus = "parallax creates clarity from multiple perspectives. parallax builds depth."
    print(f"Training Corpus ({len(corpus)} chars):")
    print(f'"{corpus}"\n')

    tokenizer = CharTokenizer(corpus)
    tokens = np.array(tokenizer.encode(corpus), dtype=np.int64)

    config = ParallaxLMConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        max_seq_len=16,
        use_rope=True,
    )
    model = ParallaxLM(config)
    optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.01)

    print("Training micro-transformer on next-token prediction...")
    batch_size = 4
    seq_len = 12

    for step in range(1, 26):
        # Sample random chunks
        idx_starts = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
        x_batch = np.stack([tokens[i : i + seq_len] for i in idx_starts])
        y_batch = np.stack([tokens[i + 1 : i + seq_len + 1] for i in idx_starts])

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = cross_entropy_loss(logits, y_batch)
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"Step {step:02d}/25 | Cross-Entropy Loss: {loss.data:7.4f}")

    print("\nSampling completion conditioned on prompt 'par':")
    prompt = tokenizer.encode("par")
    gen_tokens = generate(model, prompt, max_new_tokens=25, temperature=0.6, top_k=5)
    generated_text = tokenizer.decode(gen_tokens)
    print(f'Prompt: "par" -> Output: "{generated_text}"')


def experiment_5_kv_cache_benchmark():
    header("Experiment 5: KV Cache Speedup & Model Profiling")
    config = ParallaxLMConfig(
        vocab_size=128,
        embed_dim=48,
        num_heads=3,
        num_layers=3,
        max_seq_len=64,
        use_rope=True,
    )
    model = ParallaxLM(config)

    print("Model Architecture Profile:")
    params = count_parameters(model)
    print(f"  Embedding Params : {params['embedding']:,}")
    print(f"  Attention Params : {params['attention']:,}")
    print(f"  MLP Params       : {params['mlp']:,}")
    print(f"  Norm Params      : {params['norm']:,}")
    print(f"  LM Head Params   : {params['head']:,}")
    print(f"  Total Parameters : {params['total']:,}")

    flops = estimate_flops_per_token(config)
    print(f"  Estimated FLOPs/Token: {flops:,} FLOPs\n")

    print("Benchmarking Autoregressive Generation (16 prompt tokens -> 24 new tokens):")
    uncached, cached = benchmark_inference(model, prompt_len=16, gen_len=24, num_runs=3)

    print("\nResults:")
    print(format_benchmark_table([uncached, cached]))


def main():
    print("""
=================================================================
      PARALLAX CORE: FIRST-PRINCIPLES AI PLAYGROUND LAB
=================================================================
Choose an experiment to explore:
  1. Train a Single Neuron (w * x + b Gradient Descent)
  2. Causal Attention Matrix (Q @ K^T Softmax Heatmap)
  3. Rotary Positional Embedding (RoPE Angle Visualizer)
  4. Train Micro ParallaxLM & Generate Text
  5. KV Cache Speedup Benchmark & FLOPs Profiler
  6. Run All Experiments in Sequence
  0. Exit
""")
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        choice = input("Enter option [1-6]: ").strip()

    if choice == "1":
        experiment_1_single_neuron()
    elif choice == "2":
        experiment_2_attention_matrix()
    elif choice == "3":
        experiment_3_rope_rotation()
    elif choice == "4":
        experiment_4_train_micro_lm()
    elif choice == "5":
        experiment_5_kv_cache_benchmark()
    elif choice in ("6", "all", ""):
        experiment_1_single_neuron()
        experiment_2_attention_matrix()
        experiment_3_rope_rotation()
        experiment_4_train_micro_lm()
        experiment_5_kv_cache_benchmark()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
