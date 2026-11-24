#!/usr/bin/env python3
"""
CLI runner for computing perplexity and cross-entropy evaluation on Parallax models.

Supports evaluating GGUF (.gguf) or HuggingFace SafeTensors directories against raw text files.
"""

import argparse
import os
from parallax.eval import compute_perplexity
from parallax.gguf_loader import load_gguf_pretrained
from parallax.hf_loader import load_hf_pretrained
from parallax.generator import CharTokenizer


def eval_model_on_text(
    model_path: str,
    text_path: str,
    context_len: int = 128,
    stride: int = 64,
) -> float:
    """Evaluates model perplexity on text in a given file."""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = None
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        model, tokenizer = load_gguf_pretrained(model_path, return_tokenizer=True)
    elif os.path.isdir(model_path):
        model, tokenizer = load_hf_pretrained(model_path, return_tokenizer=True)
    else:
        raise ValueError(f"Unsupported model path or format: {model_path}")

    if tokenizer is not None:
        tokens = tokenizer.encode(text)
    else:
        # Fallback to CharTokenizer
        char_tok = CharTokenizer()
        tokens = char_tok.encode(text)

    print(f"Evaluating {len(tokens)} tokens with context_len={context_len}, stride={stride}...")
    ppl = compute_perplexity(model, tokens, max_context_len=context_len, stride=stride)
    print(f"Perplexity: {ppl:.4f}")
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on text dataset.")
    parser.add_argument("model", help="Path to .gguf file or SafeTensors directory")
    parser.add_argument("text", help="Path to evaluation text file")
    parser.add_argument("--context-len", type=int, default=128, help="Context length window")
    parser.add_argument("--stride", type=int, default=64, help="Stride between evaluation windows")

    args = parser.parse_args()
    eval_model_on_text(args.model, args.text, args.context_len, args.stride)


if __name__ == "__main__":
    main()
