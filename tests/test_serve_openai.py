import http.client
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAYLOAD = {
    "messages": [{"role": "user", "content": "First Citizen:"}],
    "max_tokens": 40,
    "temperature": 0,
}


def _check(port):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    c.request("GET", "/v1/models")
    return c.getresponse().status


def test_idle_connection_does_not_block_other_clients():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    proc = subprocess.Popen([sys.executable, "-c", f"from scripts.serve_openai import run_server; run_server(port={port})"])
    idle = None
    try:
        for _ in range(50):
            try:
                if _check(port) == 200: break
            except OSError: time.sleep(0.05)
        idle = socket.create_connection(("127.0.0.1", port))
        assert _check(port) == 200
    finally:
        if idle: idle.close()
        proc.kill()
        proc.wait()


def test_served_checkpoint_never_answers_outside_its_alphabet():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    code = f"from scripts.serve_openai import run_server; run_server(port={port}, checkpoint='data/shakespeare_model.npz')"
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stderr=subprocess.DEVNULL)
    try:
        for _ in range(200):
            if proc.poll() is not None:
                assert proc.returncode != 0
                return
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
                conn.request("POST", "/v1/chat/completions", body=json.dumps(PAYLOAD))
                break
            except OSError:
                time.sleep(0.1)
        text = json.loads(conn.getresponse().read().decode())["choices"][0]["message"]["content"]
        assert any(ch.isalpha() for ch in text), repr(text)
    finally:
        proc.kill()
        proc.wait()

