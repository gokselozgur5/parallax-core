import json
import threading
import urllib.request
import pytest
from http.server import HTTPServer
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.openai_server import OpenAIServerState, create_handler


@pytest.fixture
def running_server():
    config = ParallaxLMConfig(
        vocab_size=128,
        embed_dim=16,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        use_swiglu=False,
    )
    model = ParallaxLM(config)
    state = OpenAIServerState(model=model, model_name="test-model")
    handler = create_handler(state)

    # Pick dynamic ephemeral port
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_port

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{port}"

    server.shutdown()
    server.server_close()


def test_openai_server_models_endpoint(running_server):
    url = f"{running_server}/v1/models"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"


def test_openai_server_chat_completions_endpoint(running_server):
    url = f"{running_server}/v1/chat/completions"
    payload = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
        assert data["usage"]["completion_tokens"] == 4
