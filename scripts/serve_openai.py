#!/usr/bin/env python3
"""
CLI entrypoint to launch OpenAI-compatible HTTP serving server for ParallaxLM.
"""

import argparse
import os
from http.server import HTTPServer
from typing import Optional

from parallax.checkpoint import load_checkpoint
from parallax.gguf_loader import load_gguf_pretrained
from parallax.hf_loader import load_hf_pretrained
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.openai_server import OpenAIServerState, create_handler


def run_server(
    port: int = 8000,
    host: str = "127.0.0.1",
    checkpoint: Optional[str] = None,
    gguf: Optional[str] = None,
    hf: Optional[str] = None,
    model_name: Optional[str] = None,
) -> None:
    flags_set = sum(bool(x) for x in (checkpoint, gguf, hf))
    if flags_set > 1:
        raise ValueError("Cannot specify more than one of --checkpoint, --gguf, or --hf")

    tokenizer = None
    if checkpoint:
        model, _ = load_checkpoint(checkpoint)
        default_name = os.path.splitext(os.path.basename(checkpoint))[0]
    elif gguf:
        model, tokenizer = load_gguf_pretrained(gguf, return_tokenizer=True)
        default_name = os.path.splitext(os.path.basename(gguf))[0]
    elif hf:
        model, tokenizer = load_hf_pretrained(hf, return_tokenizer=True)
        default_name = os.path.basename(os.path.abspath(hf))
    else:
        config = ParallaxLMConfig(
            vocab_size=256,
            embed_dim=64,
            num_layers=2,
            num_heads=2,
            max_seq_len=256,
            use_swiglu=False,
        )
        model = ParallaxLM(config)
        default_name = "parallax-mini"

    resolved_model_name = model_name or default_name
    state = OpenAIServerState(model=model, model_name=resolved_model_name, tokenizer=tokenizer)
    handler_cls = create_handler(state)

    server = HTTPServer((host, port), handler_cls)
    print(f"Parallax OpenAI server listening on http://{host}:{port}/v1 (model: {resolved_model_name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallax OpenAI-compatible server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.npz) file")
    parser.add_argument("--gguf", type=str, default=None, help="Path to GGUF (.gguf) file")
    parser.add_argument("--hf", type=str, default=None, help="Path to HuggingFace model directory")
    parser.add_argument("--model-name", type=str, default=None, help="Custom model name for API responses")
    args = parser.parse_args()
    run_server(
        port=args.port,
        host=args.host,
        checkpoint=args.checkpoint,
        gguf=args.gguf,
        hf=args.hf,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
