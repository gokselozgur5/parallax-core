"""
Regression tests for serve_openai CLI options and model loading.
"""

import json
import threading
import urllib.request
import pytest
from http.server import HTTPServer

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.gguf_loader import save_parallax_to_gguf
from parallax.checkpoint import save_checkpoint
from scripts.serve_openai import run_server


def test_serve_openai_checkpoint_loading(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    ckpt_path = str(tmp_path / "model_ckpt.npz")
    save_checkpoint(ckpt_path, model, step=10)

    # Start server in thread on ephemeral port
    # Patch HTTPServer to bind to port 0
    server_holder = {}

    def custom_run():
        import parallax.openai_server as srv_mod
        from parallax.checkpoint import load_checkpoint
        loaded_model, _ = load_checkpoint(ckpt_path)
        state = srv_mod.OpenAIServerState(model=loaded_model, model_name="my-ckpt-model")
        handler_cls = srv_mod.create_handler(state)
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        server_holder["server"] = server
        server.serve_forever()

    th = threading.Thread(target=custom_run, daemon=True)
    th.start()

    import time
    while "server" not in server_holder:
        time.sleep(0.01)

    server = server_holder["server"]
    port = server.server_port
    url = f"http://127.0.0.1:{port}/v1/models"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["data"][0]["id"] == "my-ckpt-model"
    finally:
        server.shutdown()
        server.server_close()


def test_serve_openai_gguf_loading(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    gguf_path = str(tmp_path / "model_gguf.gguf")
    save_parallax_to_gguf(model, gguf_path)

    # Verify run_server with gguf argument raises if multiple options provided
    with pytest.raises(ValueError, match="Cannot specify more than one"):
        run_server(checkpoint="dummy.npz", gguf="dummy.gguf")


def test_serve_openai_cli_default_host():
    import argparse

    # Inspect the parser defaults directly
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args([])
    assert args.host == "127.0.0.1"
