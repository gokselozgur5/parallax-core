"""
Parallax WebUI Visualizer Server: Zero-dependency local web backend.

Serves the interactive Attention, KV-Cache and Token Generation Visualizer UI.
Provides the `/api/visualize_step` endpoint and helper APIs.
"""

import http.server
import json
import math
import os
import socketserver
import sys
import threading
import urllib.parse
from typing import Any, Dict, List, Optional

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from parallax.generator import CharTokenizer, sample_next_token
from parallax.loss import cross_entropy_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.tensor import no_grad

HOST = "127.0.0.1"
PORT = 8088
def _resolve_html_path() -> str:
    packaged = os.path.join(os.path.dirname(__file__), "webui", "visualizer.html")
    if os.path.exists(packaged):
        return packaged
    demo_fallback = os.path.join(REPO_ROOT, "demo", "visualizer.html")
    return demo_fallback


HTML_FILE = _resolve_html_path()
MAX_REQUEST_BODY_BYTES = 5 * 1024 * 1024  # 5 MB

CORPUS = "parallax creates clarity from multiple perspectives. parallax builds depth and understanding."


class VisualizerState:
    """Encapsulates model, tokenizer, and KV cache runtime state."""

    def __init__(self, corpus: str = CORPUS, warmup_steps: int = 40):
        self.corpus = corpus
        self.tokenizer = CharTokenizer(corpus)
        self.config = ParallaxLMConfig(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=32,
            num_heads=2,
            num_layers=2,
            max_seq_len=64,
            use_rope=True,
            tie_weights=True,
        )
        self.model = ParallaxLM(self.config)
        self.optim = AdamW(self.model.parameters(), lr=0.04, weight_decay=0.01)

        # Train a few steps so outputs are reasonable
        tokens_arr = np.array(self.tokenizer.encode(corpus), dtype=np.int64)
        seq_len = 12
        if warmup_steps > 0 and len(tokens_arr) > seq_len + 1:
            for _ in range(warmup_steps):
                starts = np.random.randint(0, len(tokens_arr) - seq_len - 1, size=4)
                xb = np.stack([tokens_arr[i : i + seq_len] for i in starts])
                yb = np.stack([tokens_arr[i + 1 : i + seq_len + 1] for i in starts])
                self.optim.zero_grad()
                logits = self.model(xb)
                loss = cross_entropy_loss(logits, yb)
                loss.backward()
                self.optim.step()

    def step_generation(
        self,
        prompt: str,
        generated_tokens: Optional[List[str]] = None,
        use_cache: bool = True,
        temperature: float = 0.8,
        top_k: Optional[int] = 5,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform a single forward pass / generation step and extract KV cache & attention state."""
        if generated_tokens is None:
            generated_tokens = []

        # Full text so far
        prompt_tokens = list(prompt) if prompt else ["P"]
        current_sequence = prompt_tokens + generated_tokens

        # Map to IDs
        token_ids = [self.tokenizer.stoi.get(ch, 0) for ch in current_sequence]
        if not token_ids:
            token_ids = [0]
            current_sequence = ["<unk>"]

        max_seq_len = self.config.max_seq_len
        # Constrain context to max_seq_len
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[-max_seq_len:]
            current_sequence = current_sequence[-max_seq_len:]

        idx = np.array([token_ids], dtype=np.int64)

        # Forward pass with no_grad to capture attention weights in blocks
        with no_grad():
            logits = self.model(idx)

        # Compute next token prediction
        next_token_logits = logits.data[0, -1, :]
        next_token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        next_token_str = self.tokenizer.decode([next_token_id])

        # Top 5 candidate probabilities
        scaled_logits = next_token_logits / max(temperature, 1e-5)
        shifted = scaled_logits - np.max(scaled_logits)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        top_indices = np.argsort(probs)[::-1][:5]
        top_candidates = [
            {
                "token": self.tokenizer.decode([int(t)]),
                "prob": float(probs[t]),
                "id": int(t),
            }
            for t in top_indices
        ]

        # Extract Attention Matrix from layer 0 (or all layers)
        # block.attn.last_attn_weights is [batch_size, num_heads, seq_len, seq_len]
        layers_attn = []
        for l_idx, block in enumerate(self.model.blocks):
            if hasattr(block.attn, "last_attn_weights") and block.attn.last_attn_weights is not None:
                # Shape: [num_heads, T, T]
                attn_weights = block.attn.last_attn_weights[0]
                num_heads = attn_weights.shape[0]
                heads = [attn_weights[h].tolist() for h in range(num_heads)]
                layers_attn.append({
                    "layer": l_idx,
                    "num_heads": num_heads,
                    "heads": heads,
                })

        # Calculate KV-Cache Memory & Block Usage
        # Each layer caches K and V: shape [B, num_kv_heads, current_len, head_dim]
        # dtype float64 = 8 bytes
        seq_len_current = len(token_ids)
        num_layers = self.config.num_layers
        num_kv_heads = self.config.num_kv_heads or self.config.num_heads
        head_dim = self.config.embed_dim // self.config.num_heads
        bytes_per_elem = 8  # float64

        kv_elements_per_token = num_layers * 2 * num_kv_heads * head_dim
        kv_bytes_current = seq_len_current * kv_elements_per_token * bytes_per_elem
        kv_bytes_max = max_seq_len * kv_elements_per_token * bytes_per_elem
        kv_percent = round((seq_len_current / max_seq_len) * 100, 2)

        return {
            "tokens": current_sequence,
            "seq_len": seq_len_current,
            "max_seq_len": max_seq_len,
            "next_token": next_token_str,
            "next_token_id": int(next_token_id),
            "top_candidates": top_candidates,
            "kv_cache": {
                "active_tokens": seq_len_current,
                "max_tokens": max_seq_len,
                "used_bytes": kv_bytes_current,
                "max_bytes": kv_bytes_max,
                "percent_used": kv_percent,
                "num_layers": num_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "use_cache": use_cache,
            },
            "attention": layers_attn,
        }


_STATE: Optional[VisualizerState] = None
_STATE_LOCK = threading.Lock()


def get_visualizer_state() -> VisualizerState:
    global _STATE
    with _STATE_LOCK:
        if _STATE is None:
            _STATE = VisualizerState()
        return _STATE


def __getattr__(name: str) -> Any:
    if name == "STATE":
        return get_visualizer_state()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class VisualizerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for Visualizer Web UI and REST API."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _get_allowed_origin(self) -> Optional[str]:
        origin = self.headers.get("Origin")
        if not origin:
            return None
        try:
            parsed = urllib.parse.urlsplit(origin)
            h = (parsed.hostname or "").lower()
            if h in ("localhost", "127.0.0.1", "::1"):
                return origin
            host_header = self.headers.get("Host", "")
            if host_header:
                server_host = host_header.split(":")[0].lower()
                if h == server_host:
                    return origin
        except Exception:
            return None
        return None

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        allowed_origin = self._get_allowed_origin()
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        allowed_origin = self._get_allowed_origin()
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/visualizer", "/visualizer.html"):
            self.serve_file(HTML_FILE, "text/html; charset=utf-8")
        elif path == "/api/info":
            state = get_visualizer_state()
            self.send_json({
                "model": "ParallaxLM-Micro",
                "vocab_size": state.config.vocab_size,
                "embed_dim": state.config.embed_dim,
                "num_heads": state.config.num_heads,
                "num_layers": state.config.num_layers,
                "max_seq_len": state.config.max_seq_len,
            })
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        raw_cl = self.headers.get("Content-Length")
        try:
            content_length = int(raw_cl) if raw_cl is not None else 0
            if content_length < 0:
                raise ValueError("Negative Content-Length")
        except (ValueError, TypeError):
            self.send_json({"error": "Invalid Content-Length header"}, 400)
            return

        if content_length > MAX_REQUEST_BODY_BYTES:
            self.send_json({"error": "Request body too large"}, 413)
            return

        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            req = json.loads(body.decode("utf-8")) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        if not isinstance(req, dict):
            self.send_json({"error": "Request body must be a JSON object"}, 400)
            return

        try:
            if parsed.path == "/api/visualize_step":
                self.handle_visualize_step(req)
            elif parsed.path == "/api/reset":
                self.send_json({"status": "ok", "message": "State reset ready"})
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"404 Not Found")
        except (ValueError, TypeError, KeyError) as e:
            self.send_json({"error": f"Invalid request parameters: {e}"}, 400)
        except Exception as e:
            self.send_json({"error": f"Internal server error: {e}"}, 500)

    def handle_visualize_step(self, req: Dict[str, Any]) -> None:
        prompt = req.get("prompt", "parallax")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        generated_tokens = req.get("generated_tokens", [])
        if not isinstance(generated_tokens, list):
            raise TypeError("generated_tokens must be a list")
        use_cache = bool(req.get("use_cache", True))
        temp_val = req.get("temperature", 0.8)
        temperature = float(temp_val)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")

        top_k = req.get("top_k", 5)
        if top_k is not None:
            top_k = int(top_k)
            if top_k <= 0:
                raise ValueError("top_k must be positive")

        top_p = req.get("top_p", None)
        if top_p is not None:
            top_p = float(top_p)
            if not math.isfinite(top_p) or top_p <= 0.0 or top_p > 1.0:
                raise ValueError("top_p must be in (0, 1]")

        state = get_visualizer_state()
        with _STATE_LOCK:
            result = state.step_generation(
                prompt=prompt,
                generated_tokens=generated_tokens,
                use_cache=use_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        self.send_json(result)

    def serve_file(self, filepath: str, content_type: str) -> None:
        if not os.path.exists(filepath):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"File not found")
            return
        with open(filepath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(port: int = PORT, host: str = HOST) -> None:
    class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    with ThreadingServer((host, port), VisualizerHandler) as server:
        print(f"[*] Parallax Visualizer Server running at http://{host}:{port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
