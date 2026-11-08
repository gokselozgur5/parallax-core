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


def test_studio_checkpoint_and_quant_apis(studio_server_url):
    # Test loading Shakespeare checkpoint
    post_data = json.dumps({"dataset": "shakespeare"}).encode()
    req = urllib.request.Request(
        f"{studio_server_url}/api/load_checkpoint",
        data=post_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    assert resp["status"] == "ok"
    assert resp["dataset"] == "shakespeare"

    # Test toggling INT8 quantization
    quant_data = json.dumps({"quantize": True}).encode()
    req2 = urllib.request.Request(
        f"{studio_server_url}/api/toggle_quant",
        data=quant_data,
        headers={"Content-Type": "application/json"},
    )
    resp2 = json.loads(urllib.request.urlopen(req2).read().decode())
    assert resp2["status"] == "ok"
    assert resp2["is_quantized"] is True

    # Test generation under INT8 mode
    gen_data = json.dumps({"prompt": "First", "max_new_tokens": 5, "use_cache": True}).encode()
    req3 = urllib.request.Request(
        f"{studio_server_url}/api/generate",
        data=gen_data,
        headers={"Content-Type": "application/json"},
    )
    resp3 = json.loads(urllib.request.urlopen(req3).read().decode())
    assert resp3["is_quantized"] is True
    assert "output_text" in resp3

    # Test disabling INT8 quantization and restoring FP64 model
    dequant_data = json.dumps({"quantize": False}).encode()
    req4 = urllib.request.Request(
        f"{studio_server_url}/api/toggle_quant",
        data=dequant_data,
        headers={"Content-Type": "application/json"},
    )
    resp4 = json.loads(urllib.request.urlopen(req4).read().decode())
    assert resp4["status"] == "ok"
    assert resp4["is_quantized"] is False

    # Generation should now indicate unquantized
    gen_data_unquant = json.dumps({"prompt": "First", "max_new_tokens": 5, "use_cache": True}).encode()
    req5 = urllib.request.Request(
        f"{studio_server_url}/api/generate",
        data=gen_data_unquant,
        headers={"Content-Type": "application/json"},
    )
    resp5 = json.loads(urllib.request.urlopen(req5).read().decode())
    assert resp5["is_quantized"] is False

    # Test toggling quant and then resetting to parallax dataset
    _ = urllib.request.urlopen(req2)
    reset_data = json.dumps({"dataset": "parallax"}).encode()
    req_reset = urllib.request.Request(
        f"{studio_server_url}/api/load_checkpoint",
        data=reset_data,
        headers={"Content-Type": "application/json"},
    )
    resp_reset = json.loads(urllib.request.urlopen(req_reset).read().decode())
    assert resp_reset["status"] == "ok"
    assert resp_reset["dataset"] == "parallax"

    resp6 = json.loads(urllib.request.urlopen(req5).read().decode())
    assert resp6["is_quantized"] is False




def test_studio_rl_apis(studio_server_url):
    # 1. Test score endpoint
    score_data = json.dumps({"text": "parallax"}).encode()
    req = urllib.request.Request(
        f"{studio_server_url}/api/rl/score",
        data=score_data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read().decode())
    assert resp["text"] == "parallax"
    assert isinstance(resp["reward"], float)

    # 2. Test preference training step
    pref_data = json.dumps({
        "chosen": "parallax builds clarity and depth",
        "rejected": "parallax creates confusion and noise",
        "steps": 2
    }).encode()
    req2 = urllib.request.Request(
        f"{studio_server_url}/api/rl/train_preference",
        data=pref_data,
        headers={"Content-Type": "application/json"},
    )
    resp2 = json.loads(urllib.request.urlopen(req2).read().decode())
    assert "losses" in resp2
    assert "reward_chosen" in resp2
    assert "reward_rejected" in resp2
    assert "margin" in resp2

    # 3. Test policy gradient step
    policy_data = json.dumps({
        "prompt": "par"
    }).encode()
    req3 = urllib.request.Request(
        f"{studio_server_url}/api/rl/policy_step",
        data=policy_data,
        headers={"Content-Type": "application/json"},
    )
    resp3 = json.loads(urllib.request.urlopen(req3).read().decode())
    assert "cand_a" in resp3
    assert "cand_b" in resp3
    assert "advantage" in resp3
    assert "winner" in resp3

    # 4. Test DPO step
    dpo_data = json.dumps({
        "chosen": "parallax builds clarity and depth",
        "rejected": "parallax creates confusion and noise",
        "steps": 2,
        "beta": 0.1
    }).encode()
    req4 = urllib.request.Request(
        f"{studio_server_url}/api/rl/dpo_step",
        data=dpo_data,
        headers={"Content-Type": "application/json"},
    )
    resp4 = json.loads(urllib.request.urlopen(req4).read().decode())
    assert resp4["status"] == "ok"
    assert "losses" in resp4
    assert "implicit_reward_chosen" in resp4
    assert "implicit_reward_rejected" in resp4
    assert "implicit_margin" in resp4



