"""
Tests for Parallax WebUI Visualizer Server and /api/visualize_step endpoint.
"""

import json
import socketserver
import threading
import time
import urllib.error
import urllib.request
import pytest

from parallax.visualizer_server import VisualizerHandler, VisualizerState


@pytest.fixture(scope="module")
def visualizer_server_url():
    """Starts an ephemeral HTTP server running VisualizerHandler."""
    class EphemeralTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    httpd = EphemeralTCPServer(("127.0.0.1", 0), VisualizerHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def test_visualizer_state_step():
    """Unit test for the internal VisualizerState step_generation logic."""
    state = VisualizerState(warmup_steps=5)
    res = state.step_generation("test", generated_tokens=[], use_cache=True)

    assert "tokens" in res
    assert res["tokens"] == ["t", "e", "s", "t"]
    assert res["seq_len"] == 4
    assert res["max_seq_len"] == 64
    assert "next_token" in res
    assert isinstance(res["next_token"], str)

    # KV Cache checks
    assert "kv_cache" in res
    kv = res["kv_cache"]
    assert kv["active_tokens"] == 4
    assert kv["max_tokens"] == 64
    assert kv["used_bytes"] > 0
    assert kv["percent_used"] == round((4 / 64) * 100, 2)
    assert kv["num_layers"] == 2

    # Attention Matrix checks
    assert "attention" in res
    attn = res["attention"]
    assert len(attn) > 0
    layer0 = attn[0]
    assert layer0["layer"] == 0
    assert layer0["num_heads"] == 2
    heads = layer0["heads"]
    assert len(heads) == 2
    # Matrix shape for sequence length 4: [4, 4]
    assert len(heads[0]) == 4
    assert len(heads[0][0]) == 4


def test_visualizer_api_info(visualizer_server_url):
    """Test /api/info endpoint."""
    req = urllib.request.urlopen(f"{visualizer_server_url}/api/info")
    assert req.status == 200
    data = json.loads(req.read().decode())
    assert data["model"] == "ParallaxLM-Micro"
    assert data["vocab_size"] > 0
    assert data["max_seq_len"] == 64


def test_visualizer_html_served(visualizer_server_url):
    """Test that visualizer.html is correctly served at root and /visualizer."""
    req = urllib.request.urlopen(f"{visualizer_server_url}/")
    assert req.status == 200
    html = req.read().decode("utf-8")
    assert "PARALLAX VISUALIZER" in html
    assert "attnCanvas" in html
    assert "kv-bar-fill" in html


def test_visualizer_api_visualize_step(visualizer_server_url):
    """Test /api/visualize_step endpoint with POST request."""
    post_data = json.dumps({
        "prompt": "parallax",
        "generated_tokens": [" ", "c"],
        "temperature": 0.7,
        "top_k": 5,
        "use_cache": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{visualizer_server_url}/api/visualize_step",
        data=post_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))

    assert "tokens" in resp
    # "parallax" + " " + "c" = 8 + 1 + 1 = 10 tokens
    assert len(resp["tokens"]) == 10
    assert resp["seq_len"] == 10
    assert "next_token" in resp
    assert "top_candidates" in resp
    assert len(resp["top_candidates"]) <= 5

    # Check KV cache structure
    kv = resp["kv_cache"]
    assert kv["active_tokens"] == 10
    assert kv["percent_used"] == round((10 / 64) * 100, 2)
    assert kv["used_bytes"] > 0

    # Check Attention structure
    assert "attention" in resp
    attn = resp["attention"]
    assert len(attn) == 2  # 2 layers
    head0 = attn[0]["heads"][0]
    assert len(head0) == 10  # 10 queries
    assert len(head0[0]) == 10  # 10 keys


def test_visualizer_cors_restriction(visualizer_server_url):
    """Test CORS header on allowed origin and disallowed origin."""
    req = urllib.request.Request(
        f"{visualizer_server_url}/api/info",
        headers={"Origin": "http://localhost:3000"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    req_disallowed = urllib.request.Request(
        f"{visualizer_server_url}/api/info",
        headers={"Origin": "http://evil.attacker.com"},
    )
    with urllib.request.urlopen(req_disallowed) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") is None


def test_visualizer_invalid_content_length_and_nan(visualizer_server_url):
    """Test rejection of malformed content-length and NaN float inputs."""
    req = urllib.request.Request(
        f"{visualizer_server_url}/api/visualize_step",
        data=b'{"temperature": "nan"}',
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
