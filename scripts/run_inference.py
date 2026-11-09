#!/usr/bin/env python3
"""
Standalone CLI Inference Runner for ParallaxLM.

Executes streaming autoregressive generation directly from GGUF binary files
or HuggingFace SafeTensors directories with zero external ML dependencies.
"""

import argparse
import os
import sys
import time

from parallax.generator import generate_stream
from parallax.gguf_loader import load_gguf_pretrained
from parallax.hf_loader import load_hf_pretrained
from parallax.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ParallaxLM First-Principles Inference CLI"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to GGUF model file (.gguf) or HuggingFace SafeTensors directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy argmax)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV cache and force full recomputation",
    )
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ParallaxLM from: {model_path} ...")
    t0 = time.time()

    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        model, tokenizer = load_gguf_pretrained(model_path, return_tokenizer=True)
    elif os.path.isdir(model_path):
        model = load_hf_pretrained(model_path)
        # Attempt to load tokenizer if present
        tok_file = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tok_file):
            tokenizer = BPETokenizer()
        else:
            tokenizer = None
    else:
        raise ValueError(f"Unsupported model format for: {model_path}")

    load_time = time.time() - t0
    print(
        f"Model loaded in {load_time:.2f}s | "
        f"Layers: {model.config.num_layers} | "
        f"Dim: {model.config.embed_dim} | "
        f"Heads: {model.config.num_heads} | "
        f"Vocab: {model.config.vocab_size}"
    )

    print("-" * 50)
    print(f"Prompt: {args.prompt}")
    print("Generation: ", end="", flush=True)

    if tokenizer is not None:
        prompt_tokens = tokenizer.encode(args.prompt)
        eos_id = getattr(tokenizer, "eos_token_id", None)
    else:
        prompt_tokens = [1, 2]
        eos_id = None

    token_count = 0
    t_start = time.time()

    for token_id in generate_stream(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        use_cache=not args.no_cache,
        eos_token_id=eos_id,
    ):
        token_count += 1
        if tokenizer is not None:
            chunk = tokenizer.decode([token_id])
            print(chunk, end="", flush=True)
        else:
            print(f"[{token_id}]", end=" ", flush=True)

    elapsed = time.time() - t_start
    tps = token_count / elapsed if elapsed > 0 else 0.0

    print("\n" + "-" * 50)
    print(f"Generated {token_count} tokens in {elapsed:.2f}s ({tps:.2f} tok/s)")


if __name__ == "__main__":
    main()
