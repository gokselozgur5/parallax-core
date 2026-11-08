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
from parallax.checkpoint import load_checkpoint
from parallax.engine import Value
from parallax.generator import CharTokenizer, generate, sample_next_token
from parallax.loss import compute_perplexity, cross_entropy_loss
from parallax.models.gpt import ParallaxLM, ParallaxLMConfig
from parallax.optim import AdamW
from parallax.quant import quantize_model
from parallax.rl import (
    RewardModel,
    bradley_terry_reward_loss,
    compute_policy_gradient_step,
    dpo_loss,
)
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

# In-memory Reward Model initialized on same config
REWARD_MODEL = RewardModel(DEMO_CONFIG)
REWARD_OPTIM = AdamW(REWARD_MODEL.parameters(), lr=0.03, weight_decay=0.01)

# Active model pointer and state
CURRENT_MODEL = DEMO_MODEL
CURRENT_CONFIG = DEMO_CONFIG
CURRENT_TOKENIZER = TOKENIZER
IS_QUANTIZED = False
CURRENT_DATASET = "parallax"

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
        elif parsed.path == "/api/load_checkpoint":
            self.handle_load_checkpoint(req)
        elif parsed.path == "/api/toggle_quant":
            self.handle_toggle_quant(req)
        elif parsed.path == "/api/rl/score":
            self.handle_rl_score(req)
        elif parsed.path == "/api/rl/train_preference":
            self.handle_rl_train_preference(req)
        elif parsed.path == "/api/rl/policy_step":
            self.handle_rl_policy_step(req)
        elif parsed.path == "/api/rl/dpo_step":
            self.handle_rl_dpo_step(req)
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
        global CURRENT_MODEL, CURRENT_CONFIG, CURRENT_TOKENIZER, IS_QUANTIZED, CURRENT_DATASET
        params = count_parameters(CURRENT_MODEL)
        flops = estimate_flops_per_token(CURRENT_CONFIG)
        self.send_json({
            "config": {
                "vocab_size": CURRENT_CONFIG.vocab_size,
                "embed_dim": CURRENT_CONFIG.embed_dim,
                "num_heads": CURRENT_CONFIG.num_heads,
                "num_kv_heads": CURRENT_CONFIG.num_kv_heads,
                "num_layers": CURRENT_CONFIG.num_layers,
                "max_seq_len": CURRENT_CONFIG.max_seq_len,
                "use_swiglu": CURRENT_CONFIG.use_swiglu,
                "tie_weights": CURRENT_CONFIG.tie_weights,
            },
            "params": params,
            "flops_per_token": flops,
            "dataset": CURRENT_DATASET,
            "is_quantized": IS_QUANTIZED,
            "vocab": CURRENT_TOKENIZER.chars,
        })

    def handle_generate(self, req: Dict[str, Any]):
        global CURRENT_MODEL, CURRENT_TOKENIZER
        prompt_text = req.get("prompt", "par")
        max_new = int(req.get("max_new_tokens", 20))
        temp = float(req.get("temperature", 0.7))
        top_k = req.get("top_k", 5)
        top_k = int(top_k) if top_k is not None else None
        top_p = req.get("top_p", 0.9)
        top_p = float(top_p) if top_p is not None else None
        use_cache = bool(req.get("use_cache", True))

        # Encode prompt using tokenizer (fallback to char ID 0 if unknown)
        prompt_ids = CURRENT_TOKENIZER.encode(prompt_text)
        if not prompt_ids:
            prompt_ids = [0]

        t0 = time.perf_counter()
        generated_ids = generate(
            CURRENT_MODEL,
            prompt_ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        gen_text = CURRENT_TOKENIZER.decode(generated_ids)
        new_tokens_count = len(generated_ids) - len(prompt_ids)
        tokens_per_sec = (new_tokens_count / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0.0

        self.send_json({
            "prompt": prompt_text,
            "output_text": gen_text,
            "new_tokens": new_tokens_count,
            "elapsed_ms": round(elapsed_ms, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "use_cache": use_cache,
            "is_quantized": IS_QUANTIZED,
        })

    def handle_load_checkpoint(self, req: Dict[str, Any]):
        global CURRENT_MODEL, CURRENT_CONFIG, CURRENT_TOKENIZER, CURRENT_DATASET, IS_QUANTIZED, REWARD_MODEL, REWARD_OPTIM, DEMO_OPTIM
        dataset = req.get("dataset", "shakespeare")

        if dataset == "shakespeare":
            ckpt_path = os.path.join(REPO_ROOT, "data", "shakespeare_model.npz")
            txt_path = os.path.join(REPO_ROOT, "data", "shakespeare.txt")
            if os.path.exists(ckpt_path) and os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    sh_text = f.read()
                CURRENT_TOKENIZER = CharTokenizer(sh_text)
                CURRENT_MODEL, step = load_checkpoint(ckpt_path)
                CURRENT_CONFIG = CURRENT_MODEL.config
                CURRENT_DATASET = "shakespeare"
                IS_QUANTIZED = False
                REWARD_MODEL = RewardModel(CURRENT_CONFIG)
                REWARD_OPTIM = AdamW(REWARD_MODEL.parameters(), lr=0.03, weight_decay=0.01)
                DEMO_OPTIM = AdamW(CURRENT_MODEL.parameters(), lr=0.04, weight_decay=0.01)
                self.send_json({"status": "ok", "dataset": "shakespeare", "step": step})
                return

        # Fallback / Reset to default Parallax MicroLM
        CURRENT_MODEL = DEMO_MODEL
        CURRENT_CONFIG = DEMO_CONFIG
        CURRENT_TOKENIZER = TOKENIZER
        CURRENT_DATASET = "parallax"
        IS_QUANTIZED = False
        REWARD_MODEL = RewardModel(DEMO_CONFIG)
        REWARD_OPTIM = AdamW(REWARD_MODEL.parameters(), lr=0.03, weight_decay=0.01)
        DEMO_OPTIM = AdamW(CURRENT_MODEL.parameters(), lr=0.04, weight_decay=0.01)
        self.send_json({"status": "ok", "dataset": "parallax", "step": 40})

    def handle_toggle_quant(self, req: Dict[str, Any]):
        global CURRENT_MODEL, IS_QUANTIZED
        enable_quant = bool(req.get("quantize", True))

        if enable_quant and not IS_QUANTIZED:
            CURRENT_MODEL = quantize_model(CURRENT_MODEL)
            IS_QUANTIZED = True
        elif not enable_quant and IS_QUANTIZED:
            # Reload from checkpoint or base model
            if CURRENT_DATASET == "shakespeare":
                ckpt_path = os.path.join(REPO_ROOT, "data", "shakespeare_model.npz")
                CURRENT_MODEL, _ = load_checkpoint(ckpt_path)
            else:
                CURRENT_MODEL = DEMO_MODEL
            IS_QUANTIZED = False

        self.send_json({"status": "ok", "is_quantized": IS_QUANTIZED})

    def handle_attention(self, req: Dict[str, Any]):
        text = req.get("text", "PARALLAX")[:16]  # Max 16 chars for responsive matrix
        if not text:
            text = "PARALLAX"

        token_ids = CURRENT_TOKENIZER.encode(text)[:16]
        if not token_ids:
            token_ids = [0]

        tokens = [CURRENT_TOKENIZER.decode([t]) for t in token_ids]
        idx = np.array([token_ids], dtype=np.int64)
        _ = CURRENT_MODEL(idx)

        first_attn = CURRENT_MODEL.blocks[0].attn
        attn_matrix = first_attn.last_attn_weights[0]  # Shape: [num_heads, seq_len, seq_len]
        num_heads = attn_matrix.shape[0]

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

    def handle_rl_score(self, req: Dict[str, Any]):
        text = req.get("text", "")
        if not text:
            self.send_json({"reward": 0.0})
            return

        ids = CURRENT_TOKENIZER.encode(text)
        if not ids:
            ids = [0]
        idx = np.array([ids], dtype=np.int64)

        reward_tensor = REWARD_MODEL(idx)
        reward_score = float(reward_tensor.data[0, 0])
        self.send_json({"text": text, "reward": round(reward_score, 3)})

    def handle_rl_train_preference(self, req: Dict[str, Any]):
        chosen_text = req.get("chosen", "parallax builds clarity and depth")
        rejected_text = req.get("rejected", "parallax creates confusion and noise")
        steps = int(req.get("steps", 5))

        chosen_ids = CURRENT_TOKENIZER.encode(chosen_text) or [0]
        rejected_ids = CURRENT_TOKENIZER.encode(rejected_text) or [0]

        # Align length with pad
        max_l = max(len(chosen_ids), len(rejected_ids))
        chosen_padded = chosen_ids + [0] * (max_l - len(chosen_ids))
        rejected_padded = rejected_ids + [0] * (max_l - len(rejected_ids))

        idx_c = np.array([chosen_padded], dtype=np.int64)
        idx_r = np.array([rejected_padded], dtype=np.int64)

        losses = []
        for _ in range(steps):
            REWARD_OPTIM.zero_grad()
            rc = REWARD_MODEL(idx_c)
            rr = REWARD_MODEL(idx_r)
            loss = bradley_terry_reward_loss(rc, rr)
            loss.backward()
            REWARD_OPTIM.step()
            losses.append(round(float(loss.data), 4))

        final_rc = float(REWARD_MODEL(idx_c).data[0, 0])
        final_rr = float(REWARD_MODEL(idx_r).data[0, 0])

        self.send_json({
            "losses": losses,
            "reward_chosen": round(final_rc, 3),
            "reward_rejected": round(final_rr, 3),
            "margin": round(final_rc - final_rr, 3),
        })

    def handle_rl_policy_step(self, req: Dict[str, Any]):
        # Model generates two responses given a prompt, scores them with RewardModel, and performs policy gradient update
        prompt = req.get("prompt", "par")
        prompt_ids = CURRENT_TOKENIZER.encode(prompt) or [0]

        # Generate candidate A
        cand_a_ids = generate(CURRENT_MODEL, prompt_ids, max_new_tokens=10, temperature=0.9, use_cache=True)
        # Generate candidate B
        cand_b_ids = generate(CURRENT_MODEL, prompt_ids, max_new_tokens=10, temperature=0.9, use_cache=True)

        text_a = CURRENT_TOKENIZER.decode(cand_a_ids)
        text_b = CURRENT_TOKENIZER.decode(cand_b_ids)

        # Score with reward model
        r_a = float(REWARD_MODEL(np.array([cand_a_ids], dtype=np.int64)).data[0, 0])
        r_b = float(REWARD_MODEL(np.array([cand_b_ids], dtype=np.int64)).data[0, 0])

        # Baseline is average reward
        baseline = (r_a + r_b) / 2.0

        # Policy update on candidate with higher reward
        DEMO_OPTIM.zero_grad()
        best_ids = cand_a_ids if r_a >= r_b else cand_b_ids
        best_reward = max(r_a, r_b)
        compute_policy_gradient_step(CURRENT_MODEL, np.array([best_ids], dtype=np.int64), reward=best_reward, baseline=baseline)
        DEMO_OPTIM.step()

        self.send_json({
            "cand_a": {"text": text_a, "reward": round(r_a, 3)},
            "cand_b": {"text": text_b, "reward": round(r_b, 3)},
            "advantage": round(best_reward - baseline, 3),
            "winner": "Candidate A" if r_a >= r_b else "Candidate B",
        })

    def handle_rl_dpo_step(self, req: Dict[str, Any]):
        chosen_text = req.get("chosen", "parallax builds clarity and depth")
        rejected_text = req.get("rejected", "parallax creates confusion and noise")
        beta = float(req.get("beta", 0.1))
        steps = int(req.get("steps", 3))

        chosen_ids = CURRENT_TOKENIZER.encode(chosen_text) or [0]
        rejected_ids = CURRENT_TOKENIZER.encode(rejected_text) or [0]

        # Align length with pad for batch forward
        max_l = max(len(chosen_ids), len(rejected_ids))
        c_padded = chosen_ids + [0] * (max_l - len(chosen_ids))
        r_padded = rejected_ids + [0] * (max_l - len(rejected_ids))

        idx_c = np.array([c_padded], dtype=np.int64)
        idx_r = np.array([r_padded], dtype=np.int64)

        # Reference model frozen outputs (using current model before DPO steps as reference)
        ref_c_logits = CURRENT_MODEL(idx_c).data.copy()
        ref_r_logits = CURRENT_MODEL(idx_r).data.copy()

        losses = []
        implicit_rw = 0.0
        implicit_rl = 0.0

        for _ in range(steps):
            DEMO_OPTIM.zero_grad()
            p_c = CURRENT_MODEL(idx_c)
            p_r = CURRENT_MODEL(idx_r)

            loss, implicit_rw, implicit_rl = dpo_loss(
                policy_chosen_logits=p_c,
                policy_rejected_logits=p_r,
                ref_chosen_logits=ref_c_logits,
                ref_rejected_logits=ref_r_logits,
                chosen_targets=idx_c,
                rejected_targets=idx_r,
                beta=beta,
            )
            DEMO_OPTIM.step()
            losses.append(round(loss, 4))

        margin = implicit_rw - implicit_rl
        self.send_json({
            "status": "ok",
            "losses": losses,
            "implicit_reward_chosen": round(implicit_rw, 3),
            "implicit_reward_rejected": round(implicit_rl, 3),
            "implicit_margin": round(margin, 3),
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
