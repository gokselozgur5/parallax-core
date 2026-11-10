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


DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024  # 10MB


class OpenAIServerState:
    def __init__(
        self,
        model: ParallaxLM,
        model_name: str = "parallax-lm",
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_tokens_limit: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.max_request_bytes = max_request_bytes
        # Context limit fallback from model config or custom limit
        model_ctx = getattr(model.config, "max_seq_len", None)
        self.max_tokens_limit = max_tokens_limit if max_tokens_limit is not None else model_ctx
        self.tokenizer = tokenizer
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

        def _send_sse_chunk(self, data: Dict[str, Any]) -> None:
            chunk = f"data: {json.dumps(data)}\n\n".encode("utf-8")
            self.wfile.write(chunk)
            self.wfile.flush()

        def do_POST(self) -> None:
            if self.path == "/v1/chat/completions":
                try:
                    content_length_header = self.headers.get("Content-Length")
                    try:
                        content_length = int(content_length_header) if content_length_header is not None else 0
                    except ValueError:
                        self._send_json(400, {"error": {"message": "Invalid Content-Length header", "type": "invalid_request_error"}})
                        return

                    if content_length < 0:
                        self._send_json(400, {"error": {"message": "Negative Content-Length not allowed", "type": "invalid_request_error"}})
                        return

                    if content_length > state.max_request_bytes:
                        self._send_json(413, {"error": {"message": "Request entity too large", "type": "invalid_request_error"}})
                        return

                    body = self.rfile.read(content_length)
                    data = json.loads(body.decode("utf-8"))

                    messages = data.get("messages", [])
                    max_tokens = int(data.get("max_tokens", 32))
                    if state.max_tokens_limit is not None:
                        max_tokens = min(max_tokens, state.max_tokens_limit)
                    temperature = float(data.get("temperature", 0.0))
                    stream = bool(data.get("stream", False))

                    prompt_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
                    if state.tokenizer is not None:
                        prompt_tokens = state.tokenizer.encode(prompt_text)
                    else:
                        prompt_tokens = [ord(c) % state.model.config.vocab_size for c in prompt_text]
                    if not prompt_tokens:
                        prompt_tokens = [0]

                    eos_token_id = None
                    if state.tokenizer is not None and getattr(state.tokenizer, "eos_token_id", None) is not None:
                        eos_token_id = state.tokenizer.eos_token_id

                    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                    req = SequenceRequest(
                        request_id=req_id,
                        prompt_tokens=prompt_tokens,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        eos_token_id=eos_token_id,
                    )
                    state.engine.add_request(req)

                    def decode_tokens(tokens: List[int]) -> str:
                        if state.tokenizer is not None:
                            return state.tokenizer.decode(tokens)
                        return "".join(chr(t if 32 <= t <= 126 else 32) for t in tokens)

                    if stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()

                        # Emit initial chunk with role: assistant
                        initial_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": state.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        self._send_sse_chunk(initial_payload)

                        finish_reason = "length" if max_tokens == 0 else "stop"
                        last_tok = None
                        if max_tokens > 0:
                            while req.status != RequestStatus.FINISHED:
                                step_outputs = state.engine.step()
                                for r_id, tok in step_outputs:
                                    if r_id == req_id:
                                        last_tok = tok
                                        piece = decode_tokens([tok])
                                        chunk_payload = {
                                            "id": req_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": state.model_name,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": piece},
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        self._send_sse_chunk(chunk_payload)

                            eos_id = getattr(req, "eos_token_id", None)
                            if eos_id is not None and last_tok == eos_id:
                                finish_reason = "stop"
                            else:
                                finish_reason = "length"

                        # Send final chunk with finish_reason
                        stop_payload = {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": state.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                        self._send_sse_chunk(stop_payload)
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return

                    # Non-streaming execution
                    if max_tokens > 0:
                        while req.status != RequestStatus.FINISHED:
                            state.engine.step()

                    # Convert generated tokens back to text
                    gen_text = decode_tokens(req.generated_tokens) if req.generated_tokens else ""

                    eos_id = getattr(req, "eos_token_id", None)
                    if max_tokens == 0:
                        finish_reason = "length"
                    elif eos_id is not None and req.generated_tokens and req.generated_tokens[-1] == eos_id:
                        finish_reason = "stop"
                    else:
                        finish_reason = "length"

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
                                "finish_reason": finish_reason,
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
