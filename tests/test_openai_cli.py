"""
Regression tests for serve_openai CLI options and model loading.
"""

import http.server
import json
import threading
import time
import unittest.mock
import urllib.request
import pytest

from parallax.checkpoint import save_checkpoint
from parallax.gguf_loader import save_parallax_to_gguf
from parallax.hf_loader import save_hf_pretrained
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from scripts.serve_openai import main, run_server


def _spin_server(kwargs):
    captured = {}
    orig_init = http.server.ThreadingHTTPServer.__init__

    def patched_init(self, addr, handler_cls, bind_and_activate=True):
        orig_init(self, addr, handler_cls, bind_and_activate)
        captured["server"] = self

    with unittest.mock.patch("http.server.ThreadingHTTPServer.__init__", patched_init):
        thread = threading.Thread(target=run_server, kwargs=dict(port=0, **kwargs), daemon=True)
        thread.start()
        while "server" not in captured:
            time.sleep(0.01)

    server = captured["server"]
    port = server.server_port
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", method="GET")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["id"]
    finally:
        server.shutdown()
        server.server_close()


def test_serve_openai_checkpoint_loading(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=2,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    ckpt_path = str(tmp_path / "model_ckpt.npz")
    vocab_path = str(tmp_path / "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("ab")
    save_checkpoint(ckpt_path, model, step=10)

    def custom_run():
        return _spin_server({"checkpoint": ckpt_path, "vocab": vocab_path, "model_name": "my-ckpt-model"})

    model_id = custom_run()
    assert model_id == "my-ckpt-model"


def test_serve_openai_gguf_loading(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
        use_swiglu=True,
    )
    model = ParallaxLM(config)
    gguf_path = str(tmp_path / "model_gguf.gguf")
    save_parallax_to_gguf(model, gguf_path)

    # Verify run_server with gguf argument raises if multiple options provided
    with pytest.raises(ValueError, match="Cannot specify more than one"):
        run_server(checkpoint="dummy.npz", gguf="dummy.gguf")

    model_id = _spin_server({"gguf": gguf_path})
    assert model_id == "model_gguf"


def test_serve_openai_hf_loading(tmp_path):
    config = ParallaxLMConfig(
        vocab_size=64,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=32,
        use_swiglu=True,
    )
    model = ParallaxLM(config)
    hf_dir = str(tmp_path / "model_hf")
    save_hf_pretrained(model, hf_dir)

    model_id = _spin_server({"hf": hf_dir})
    assert model_id == "model_hf"


def test_serve_openai_cli_default_host():
    with unittest.mock.patch("sys.argv", ["serve_openai.py"]):
        with unittest.mock.patch("parallax.cli.serve_openai.run_server") as mock_run:
            main()
            assert mock_run.call_args.kwargs["host"] == "127.0.0.1"
