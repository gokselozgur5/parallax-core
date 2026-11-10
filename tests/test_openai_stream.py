import json
import threading
import urllib.request
import pytest
from http.server import HTTPServer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.openai_server import OpenAIServerState, create_handler


@pytest.fixture
def running_stream_server():
    config = ParallaxLMConfig(
        vocab_size=128,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    state = OpenAIServerState(model=model, model_name="stream-model")
    handler = create_handler(state)

    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_port

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()


def test_openai_server_chat_completions_sse_streaming(running_stream_server):
    url = f"{running_stream_server}/v1/chat/completions"
    payload = json.dumps({
        "model": "stream-model",
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

        raw_lines = resp.read().decode("utf-8").strip().split("\n\n")
        assert len(raw_lines) >= 2

        chunks = []
        for block in raw_lines:
            if block.startswith("data: "):
                data_str = block[6:]
                if data_str == "[DONE]":
                    break
                parsed = json.loads(data_str)
                chunks.append(parsed)

        assert len(chunks) >= 5
        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"
            assert "choices" in chunk

        # Initial chunk should have role: assistant
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        # Last chunk before [DONE] should have finish_reason length or stop
        assert chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop")


def test_openai_server_streaming_max_tokens_zero(running_stream_server):
    url = f"{running_stream_server}/v1/chat/completions"
    payload = json.dumps({
        "model": "stream-model",
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 0,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        raw_lines = resp.read().decode("utf-8").strip().split("\n\n")
        chunks = []
        for block in raw_lines:
            if block.startswith("data: "):
                data_str = block[6:]
                if data_str == "[DONE]":
                    break
                chunks.append(json.loads(data_str))

        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["finish_reason"] == "length"
