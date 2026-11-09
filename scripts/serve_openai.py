#!/usr/bin/env python3
"""
CLI entrypoint to launch OpenAI-compatible HTTP serving server for ParallaxLM.
"""

import argparse
from http.server import HTTPServer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.openai_server import OpenAIServerState, create_handler


def run_server(port: int = 8000, host: str = "0.0.0.0") -> None:
    config = ParallaxLMConfig(
        vocab_size=256,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=256,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    state = OpenAIServerState(model=model, model_name="parallax-mini")
    handler_cls = create_handler(state)

    server = HTTPServer((host, port), handler_cls)
    print(f"Parallax OpenAI server listening on http://{host}:{port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallax OpenAI-compatible server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
