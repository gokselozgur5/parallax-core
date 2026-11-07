"""
Parallax Studio Server: Zero-dependency local web backend.

Serves the interactive Parallax Studio UI and exposes first-principles REST APIs for:
1. Autoregressive Text Generation with live token step timings & KV Cache toggle
2. Causal Attention Matrix computation for all attention heads
3. Single Neuron / Micro Autograd DAG visualization via Mermaid
4. In-memory Micro ParallaxLM training iterations with live loss progression
"""

import http.server
import json
import os
import socketserver
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional

# Add repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from parallax.attention import CausalSelfAttention, apply_rope
from parallax.bench import count_parameters, estimate_flops_per_token
from parallax.engine import Value
from parallax.generator import CharTokenizer, generate, sample_next_token
from parallax.loss import compute_perplexity, cross_entropy_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.tensor import Tensor
from parallax.viz import to_mermaid

PORT = 8080
STATIC_DIR = os.path.join(os.path.dirname(__file__), "web")

# Pre-train a tiny micro-model in-memory so the UI is ready to generate immediately
CORPUS = "parallax creates clarity from multiple perspectives. parallax builds depth and understanding."
TOKENIZER = CharTokenizer(CORPUS)

DEMO_CONFIG = ParallaxLMConfig(
    vocab_size=TOKENIZER.vocab_size,
    embed_dim=32,
    num_heads=2,
    num_layers=2,
    max_seq_len=64,
    use_rope=True,
    tie_weights=True,
)
DEMO_MODEL = ParallaxLM(DEMO_CONFIG)
DEMO_OPTIM = AdamW(DEMO_MODEL.parameters(), lr=0.04, weight_decay=0.01)

# Perform quick initial warmup training
tokens_arr = np.array(TOKENIZER.encode(CORPUS), dtype=np.int64)
seq_l = 12
for _ in range(40):
    starts = np.random.randint(0, len(tokens_arr) - seq_l - 1, size=4)
    xb = np.stack([tokens_arr[i : i + seq_l] for i in starts])
    yb = np.stack([tokens_arr[i + 1 : i + seq_l + 1] for i in starts])
    DEMO_OPTIM.zero_grad()
    logits = DEMO_MODEL(xb)
    loss = cross_entropy_loss(logits, yb)
    loss.backward()
    DEMO_OPTIM.step()


class StudioHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            req = json.loads(body)
        except Exception:
            req = {}

        if parsed.path == "/api/generate":
            self.handle_generate(req)
        elif parsed.path == "/api/attention":
            self.handle_attention(req)
        elif parsed.path == "/api/dag":
            self.handle_dag(req)
        elif parsed.path == "/api/train_step":
            self.handle_train_step(req)
        elif parsed.path == "/api/model_info":
            self.handle_model_info()
        else:
            self.send_error(404, "Endpoint not found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/model_info":
            self.handle_model_info()
        else:
            super().do_GET()

    def send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def handle_model_info(self):
        params = count_parameters(DEMO_MODEL)
        flops = estimate_flops_per_token(DEMO_CONFIG)
        self.send_json({
            "config": {
                "vocab_size": DEMO_CONFIG.vocab_size,
                "embed_dim": DEMO_CONFIG.embed_dim,
                "num_heads": DEMO_CONFIG.num_heads,
                "num_layers": DEMO_CONFIG.num_layers,
                "max_seq_len": DEMO_CONFIG.max_seq_len,
                "tie_weights": DEMO_CONFIG.tie_weights,
            },
            "params": params,
            "flops_per_token": flops,
            "corpus": CORPUS,
            "vocab": TOKENIZER.chars,
        })

    def handle_generate(self, req: Dict[str, Any]):
        prompt_text = req.get("prompt", "par")
        max_new = int(req.get("max_new_tokens", 20))
        temp = float(req.get("temperature", 0.7))
        top_k = req.get("top_k", 5)
        top_k = int(top_k) if top_k is not None else None
        top_p = req.get("top_p", 0.9)
        top_p = float(top_p) if top_p is not None else None
        use_cache = bool(req.get("use_cache", True))

        # Encode prompt using tokenizer (fallback to char ID 0 if unknown)
        prompt_ids = TOKENIZER.encode(prompt_text)
        if not prompt_ids:
            prompt_ids = [0]

        t0 = time.perf_counter()
        generated_ids = generate(
            DEMO_MODEL,
            prompt_ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gen_text = TOKENIZER.decode(generated_ids)
        new_tokens_count = len(generated_ids) - len(prompt_ids)
        tokens_per_sec = (new_tokens_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

        self.send_json({
            "prompt": prompt_text,
            "output_text": gen_text,
            "new_tokens": new_tokens_count,
            "elapsed_ms": round(elapsed_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "use_cache": use_cache,
        })

    def handle_attention(self, req: Dict[str, Any]):
        text = req.get("text", "PARALLAX")[:16]  # Max 16 chars for responsive matrix
        if not text:
            text = "PARALLAX"

        tokens = list(text)
        seq_len = len(tokens)
        embed_dim = 16
        num_heads = 2
        head_dim = embed_dim // num_heads

        # Forward pass on standalone attention block to capture matrix
        mha = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, use_rope=True)
        # Create deterministic pseudo-embeddings from char ordinals
        x_data = np.zeros((1, seq_len, embed_dim), dtype=np.float64)
        for i, ch in enumerate(text):
            x_data[0, i, :] = np.sin(np.arange(embed_dim) * (ord(ch) + i + 1) * 0.1)

        _ = mha(Tensor(x_data))
        attn_matrix = mha.last_attn_weights[0]  # Shape: [num_heads, seq_len, seq_len]

        heads_data = []
        for h in range(num_heads):
            heads_data.append(attn_matrix[h].tolist())

        self.send_json({
            "tokens": tokens,
            "num_heads": num_heads,
            "attention_heads": heads_data,
        })

    def handle_dag(self, req: Dict[str, Any]):
        expr_type = req.get("type", "scalar")
        w_val = float(req.get("w", 2.0))
        x_val = float(req.get("x", 3.0))
        b_val = float(req.get("b", -1.0))

        if expr_type == "neuron":
            w = Value(w_val, label="w")
            x = Value(x_val, label="x")
            b = Value(b_val, label="b")
            wx = w * x
            wx.label = "w*x"
            y = wx + b
            y.label = "y"
            target = Value(10.0, label="target")
            diff = y - target
            diff.label = "diff"
            loss = diff ** 2
            loss.label = "loss"
            loss.backward()
            mermaid_code = to_mermaid(loss)
            result_val = loss.data
        else:
            a = Value(w_val, label="a")
            b = Value(x_val, label="b")
            c = a + b
            c.label = "c"
            d = c * b_val
            d.label = "d"
            d.backward()
            mermaid_code = to_mermaid(d)
            result_val = d.data

        self.send_json({
            "mermaid": mermaid_code,
            "result": result_val,
        })

    def handle_train_step(self, req: Dict[str, Any]):
        num_steps = int(req.get("steps", 5))
        lr = float(req.get("lr", 0.04))
        DEMO_OPTIM.lr = lr

        losses = []
        for _ in range(num_steps):
            starts = np.random.randint(0, len(tokens_arr) - seq_l - 1, size=4)
            xb = np.stack([tokens_arr[i : i + seq_l] for i in starts])
            yb = np.stack([tokens_arr[i + 1 : i + seq_l + 1] for i in starts])
            DEMO_OPTIM.zero_grad()
            logits = DEMO_MODEL(xb)
            loss = cross_entropy_loss(logits, yb)
            loss.backward()
            DEMO_OPTIM.step()
            losses.append(round(float(loss.data), 4))

        ppl = compute_perplexity(losses[-1]) if losses else 0.0
        self.send_json({
            "losses": losses,
            "last_loss": losses[-1] if losses else 0.0,
            "perplexity": round(ppl, 2),
            "step_count": DEMO_OPTIM.step_count,
        })


def run_server(port: int = PORT):
    os.makedirs(STATIC_DIR, exist_ok=True)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), StudioHandler) as httpd:
        print(f"Parallax Studio running at http://localhost:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    run_server(p)
