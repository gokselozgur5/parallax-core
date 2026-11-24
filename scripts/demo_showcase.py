"""Showcase demo for Parallax v0.2.0: Autograd, Memory Scaling, Dispatch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from parallax.backends.dispatch import candidate_backends, dispatch
from parallax.tensor import Tensor


def verify_autograd_gradients() -> dict[str, float]:
    """Verify backprop gradients match finite differences (error < 1e-4)."""
    x_val = np.array([[-1.5, 0.5], [2.0, -0.5]], dtype=np.float64)
    w_val = np.array([[0.8, -0.2], [-0.4, 1.2]], dtype=np.float64)

    def func(xa: np.ndarray, wa: np.ndarray) -> np.ndarray:
        return np.sum((xa @ wa) ** 2 + np.maximum(0.0, xa))

    x_t, w_t = Tensor(x_val.copy(), requires_grad=True), Tensor(w_val.copy(), requires_grad=True)
    ((x_t @ w_t) * (x_t @ w_t) + x_t.relu()).sum().backward()

    eps = 1e-6
    gx, gw = np.zeros_like(x_val), np.zeros_like(w_val)
    for i in range(x_val.shape[0]):
        for j in range(x_val.shape[1]):
            xp, xm = x_val.copy(), x_val.copy()
            xp[i, j] += eps; xm[i, j] -= eps
            gx[i, j] = (func(xp, w_val) - func(xm, w_val)) / (2 * eps)
            wp, wm = w_val.copy(), w_val.copy()
            wp[i, j] += eps; wm[i, j] -= eps
            gw[i, j] = (func(x_val, wp) - func(x_val, wm)) / (2 * eps)

    ex = float(np.max(np.abs(np.array(x_t.grad) - gx)))
    ew = float(np.max(np.abs(np.array(w_t.grad) - gw)))
    return {"max_grad_error": max(ex, ew), "err_x": ex, "err_w": ew}


def benchmark_memory_scaling(seq_lens: tuple[int, ...] = (256, 1024, 4096)) -> list[dict[str, Any]]:
    """Compare Transformer KV-cache O(N) vs BitMambaLM O(1) recurrent memory."""
    layers, heads, hdim, dbytes = 2, 4, 16, 4
    mamba_bytes = (layers * (128 * 16 + 128 * 4) * dbytes)
    return [
        {
            "seq_len": s,
            "transformer_kv_bytes": 2 * layers * heads * s * hdim * dbytes,
            "mamba_state_bytes": mamba_bytes,
            "ratio_transformer_over_mamba": round((2 * layers * heads * s * hdim * dbytes) / mamba_bytes, 2),
        }
        for s in seq_lens
    ]


def benchmark_dispatch_timing(quick: bool = True) -> dict[str, Any]:
    """Measure dispatch timing for GEMM and RMSNorm."""
    dim = 256 if quick else 512
    a, b = np.random.randn(dim, dim).astype(np.float32), np.random.randn(dim, dim).astype(np.float32)
    w = np.ones((dim,), dtype=np.float32)
    iters = 10 if quick else 50
    _ = dispatch("gemm", a, b); _ = dispatch("rmsnorm", a, w, eps=1e-5)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = dispatch("gemm", a, b)
    t_gemm = ((time.perf_counter() - t0) / iters) * 1000.0
    t1 = time.perf_counter()
    for _ in range(iters):
        _ = dispatch("rmsnorm", a, w, eps=1e-5)
    t_rms = ((time.perf_counter() - t1) / iters) * 1000.0
    gc, rc = candidate_backends("gemm", dim * dim), candidate_backends("rmsnorm", dim * dim)
    return {
        "active_gemm_backend": gc[0] if gc else "numpy",
        "active_rmsnorm_backend": rc[0] if rc else "numpy",
        "gemm_latency_ms": round(t_gemm, 3),
        "rmsnorm_latency_ms": round(t_rms, 3),
        "matrix_dim": dim,
        "iterations": iters,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for parallax-demo showcase."""
    parser = argparse.ArgumentParser(description="Parallax v0.2.0 Showcase Demo")
    parser.add_argument("--quick", action="store_true", default=True, help="Fast mode (< 5s)")
    parser.add_argument("--full", action="store_true", help="Full benchmark mode")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    gm = verify_autograd_gradients()
    mem = benchmark_memory_scaling()
    disp = benchmark_dispatch_timing(quick=not args.full)
    ok = gm["max_grad_error"] < 1e-4

    payload = {"version": "0.2.0", "autograd": {"status": "PASSED" if ok else "FAILED", "metrics": gm}, "memory_scaling": mem, "dispatch": disp}
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if ok else 1

    print("=================================================================\n                PARALLAX v0.2.0 SHOWCASE DEMO\n=================================================================\n")
    print(f"[1] Autograd: Max error = {gm['max_grad_error']:.2e} ({'PASS' if ok else 'FAIL'})")
    print("[2] Memory: Transformer KV-Cache O(N) vs BitMambaLM O(1)")
    for r in mem:
        print(f"    seq={r['seq_len']:>4}: KV={r['transformer_kv_bytes']//1024:>5}KB | Mamba={r['mamba_state_bytes']//1024:>2}KB | Ratio={r['ratio_transformer_over_mamba']:>6.1f}x")
    print(f"[3] Dispatch: GEMM={disp['active_gemm_backend']} ({disp['gemm_latency_ms']}ms), RMSNorm={disp['active_rmsnorm_backend']} ({disp['rmsnorm_latency_ms']}ms)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
