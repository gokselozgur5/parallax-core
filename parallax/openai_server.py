#!/usr/bin/env python3
"""
OpenAI-compatible HTTP serving server for ParallaxLM.

Zero external dependencies (pure Python standard library http.server),
supporting /v1/models and /v1/chat/completions with ContinuousBatchEngine.
"""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from parallax.batching import ContinuousBatchEngine, RequestStatus, SequenceRequest
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig


class OpenAIServerState:
    def __init__(self, model: ParallaxLM, model_name: str = "parallax-lm"):
        self.model = model
        self.model_name = model_name
        self.engine = ContinuousBatchEngine(model=model, max_batch_size=8)


def create_handler(state: OpenAIServerState):
    class OpenAIRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model_name,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "parallax",
                            }
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": {"message": "Not Found", "type": "invalid_request_error"}})

        def do_POST(self) -> None:
            if self.path == "/v1/chat/completions":
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode("utf-8"))

                    messages = data.get("messages", [])
                    max_tokens = int(data.get("max_tokens", 32))
                    temperature = float(data.get("temperature", 0.0))

                    # Simple tokenization: ASCII char codes
                    prompt_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
                    prompt_tokens = [ord(c) % state.model.config.vocab_size for c in prompt_text]
                    if not prompt_tokens:
                        prompt_tokens = [0]

                    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    req = SequenceRequest(
                        request_id=req_id,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )
                    state.engine.add_request(req)

                    # Step engine until this request finishes
                    while req.status != RequestStatus.FINISHED:
                        state.engine.step()

                    # Convert generated tokens back to characters
                    gen_text = "".join(chr(t if 32 <= t <= 126 else 32) for t in req.generated_tokens)

                    response_payload = {
                        "id": req_id,
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": state.model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": gen_text,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(prompt_tokens),
                            "completion_tokens": len(req.generated_tokens),
                            "total_tokens": len(prompt_tokens) + len(req.generated_tokens),
                        },
                    }
                    self._send_json(200, response_payload)

                except Exception as e:
                    self._send_json(500, {"error": {"message": str(e), "type": "server_error"}})
            else:
                self._send_json(404, {"error": {"message": "Not Found", "type": "invalid_request_error"}})

        def log_message(self, format: str, *args: Any) -> None:
            # Suppress default noisy access logging during tests
            pass

    return OpenAIRequestHandler
