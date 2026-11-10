"""
Tests for Studio Server security, input validation, and import side-effects.

Covers:
- Issue #262: Bind to 127.0.0.1, restrict CORS headers, reject simple CORS text/plain POSTs.
- Issue #263: Structured JSON errors, request body limits (413), parameter clamping.
- Issue #264: Zero import-time model training side-effects, isolated StudioState.
"""

import json
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import pytest

from demo.studio_server import (
    HOST,
    MAX_NEW_TOKENS,
    MAX_REQUEST_BODY_BYTES,
    MAX_STEPS,
    PORT,
    StudioHandler,
    StudioState,
    run_server,
)


@pytest.fixture(scope="module")
def studio_server_batch6():
    port = 8391
    t = threading.Thread(target=run_server, args=(port, "127.0.0.1"), daemon=True)
    t.start()
    time.sleep(0.5)
    return f"http://127.0.0.1:{port}"


def test_import_has_no_side_effects_or_training():
    code = (
        "import sys\n"
        "import demo.studio_server as ss\n"
        "assert not hasattr(ss, '_state'), 'import must not initialize _state'\n"
        "assert '_state' not in sys.modules['demo.studio_server'].__dict__\n"
        "print('CLEAN_IMPORT')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "CLEAN_IMPORT" in result.stdout


def test_isolated_studio_state():
    state1 = StudioState(warmup_steps=0)
    state2 = StudioState(warmup_steps=0)
    assert state1 is not state2
    assert state1.demo_model is not state2.demo_model
    assert state1.demo_optim is not state2.demo_optim


def test_default_host_and_port():
    assert HOST == "127.0.0.1"
    assert PORT == 8080


def test_cors_same_origin_and_localhost(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/model_info",
        headers={"Origin": "http://localhost:8080"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:8080"
        assert "Origin" in resp.headers.get("Vary", "")


def test_cors_disallows_arbitrary_cross_origin(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/model_info",
        headers={"Origin": "http://evil-attacker.com"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") is None


def test_reject_unsupported_media_type_text_plain(studio_server_batch6):
    data = b'{"prompt": "hello"}'
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/generate",
        data=data,
        headers={"Content-Type": "text/plain"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 415
    body = json.loads(exc_info.value.read().decode())
    assert "error" in body
    assert "Unsupported media type" in body["error"]


def test_reject_oversized_payload(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/generate",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(MAX_REQUEST_BODY_BYTES + 1024),
        },
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 413
    body = json.loads(exc_info.value.read().decode())
    assert "Request body exceeds" in body["error"]


def test_malformed_json_returns_400(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/generate",
        data=b"{not_valid_json",
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    body = json.loads(exc_info.value.read().decode())
    assert "Invalid JSON payload" in body["error"]


def test_invalid_parameter_types_returns_structured_400(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/generate",
        data=json.dumps({"top_k": "not_an_int"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    body = json.loads(exc_info.value.read().decode())
    assert "error" in body


def test_generate_clamps_max_new_tokens(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/generate",
        data=json.dumps({"prompt": "p", "max_new_tokens": 100000}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    assert data["new_tokens"] <= MAX_NEW_TOKENS


def test_train_step_clamps_steps(studio_server_batch6):
    req = urllib.request.Request(
        f"{studio_server_batch6}/api/train_step",
        data=json.dumps({"steps": 50000, "lr": 0.01}).encode(),
        headers={"Content-Type": "application/json"},
    )
    req_zero = urllib.request.Request(
        f"{studio_server_batch6}/api/train_step",
        data=json.dumps({"steps": 0, "lr": 0.01}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_zero) as resp:
        data = json.loads(resp.read().decode())
    assert len(data["losses"]) == 1
