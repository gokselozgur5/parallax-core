"""
Integration tests for Parallax Studio Web Server and APIs.
"""

import json
import threading
import time
import urllib.request
import pytest

from demo.studio_server import run_server


@pytest.fixture(scope="module")
def studio_server_url():
    test_port = 8299
    t = threading.Thread(target=run_server, args=(test_port,), daemon=True)
    t.start()
    time.sleep(0.5)
    return f"http://localhost:{test_port}"


def test_studio_model_info_api(studio_server_url):
    req = urllib.request.urlopen(f"{studio_server_url}/api/model_info")
    data = json.loads(req.read().decode())
    assert "params" in data
    assert "config" in data
    assert data["params"]["total"] > 0
    assert data["config"]["vocab_size"] > 0


def test_studio_generate_api(studio_server_url):
    post_data = json.dumps({
        "prompt": "par",
        "max_new_tokens": 8,
        "temperature": 0.5,
        "top_k": 3,
        "use_cache": True,
    }).encode()
    req = urllib.request.Request(
        f"{studio_server_url}/api/generate",
        data=post_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    assert "output_text" in resp
    assert resp["prompt"] == "par"
    assert resp["new_tokens"] == 8
    assert resp["elapsed_ms"] > 0.0


def test_studio_attention_matrix_api(studio_server_url):
    post_data = json.dumps({"text": "TEST"}).encode()
    req = urllib.request.Request(
        f"{studio_server_url}/api/attention",
        data=post_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    assert resp["tokens"] == ["T", "E", "S", "T"]
    assert len(resp["attention_heads"]) == 2
    # Verify shape of matrix
    assert len(resp["attention_heads"][0]) == 4
    assert len(resp["attention_heads"][0][0]) == 4


def test_studio_dag_api(studio_server_url):
    post_data = json.dumps({"type": "neuron", "w": 1.0, "x": 2.0, "b": 0.5}).encode()
    req = urllib.request.Request(
        f"{studio_server_url}/api/dag",
        data=post_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    assert "mermaid" in resp
    assert "graph LR" in resp["mermaid"]


def test_studio_train_step_api(studio_server_url):
    post_data = json.dumps({"steps": 2, "lr": 0.02}).encode()
    req = urllib.request.Request(
        f"{studio_server_url}/api/train_step",
        data=post_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    assert len(resp["losses"]) == 2
    assert "perplexity" in resp
    assert resp["step_count"] > 0
