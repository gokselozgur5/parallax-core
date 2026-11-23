import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.openai_server import OpenAIServerState, create_handler

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


def _count_content_chunks(port, counts, i):
    payload = {"messages": [{"role": "user", "content": f"p{i}"}], "max_tokens": 48, "stream": True}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request("POST", "/v1/chat/completions", body=json.dumps(payload))
    lines = conn.getresponse().read().decode().split("\n")
    events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
    counts[i] = sum(1 for e in events if e["choices"][0]["delta"].get("content"))


def test_concurrent_streams_receive_every_token():
    cfg = ParallaxLMConfig(vocab_size=128, embed_dim=32, num_layers=2, num_heads=2, max_seq_len=512)
    state = OpenAIServerState(model=ParallaxLM(cfg))
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    counts = {}
    threads = [
        threading.Thread(target=_count_content_chunks, args=(server.server_port, counts, i))
        for i in range(4)
    ]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        server.shutdown()
        server.server_close()
    assert counts == {0: 48, 1: 48, 2: 48, 3: 48}

