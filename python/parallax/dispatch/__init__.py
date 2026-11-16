"""Dispatch modules for hardware-accelerated and optimized neural network layers."""

from python.parallax.dispatch.axial_rope import (
    AxialRoPEFrequencyTable,
    apply_axial_rope,
    apply_axial_rope_grid,
    dispatch_axial_rope,
    precompute_axial_freqs,
    precompute_axial_freqs_cis,
)
from python.parallax.dispatch.conv2d import (
    DispatchedConv2d,
    conv2d_backward,
    conv2d_forward,
    dispatched_conv2d,
)
from python.parallax.dispatch.kv_cache import (
    KVCacheBuffer,
    KVCacheManager,
    create_kv_cache,
)
from python.parallax.dispatch.rope import (
    RoPEFrequencyTable,
    apply_rotary_emb,
    dispatch_rope,
    precompute_freqs,
    precompute_freqs_cis,
    rotate_half,
)
from python.parallax.dispatch.sliding_window import (
    SlidingWindowAttention,
    build_sliding_window_mask,
    dispatched_sliding_window_attention,
    flash_sliding_window_tile,
    get_sliding_window_bounds,
    sliding_window_attention,
)

__all__ = [
    "AxialRoPEFrequencyTable",
    "DispatchedConv2d",
    "KVCacheBuffer",
    "KVCacheManager",
    "RoPEFrequencyTable",
    "SlidingWindowAttention",
    "apply_axial_rope",
    "apply_axial_rope_grid",
    "apply_rotary_emb",
    "build_sliding_window_mask",
    "conv2d_backward",
    "conv2d_forward",
    "create_kv_cache",
    "dispatch_axial_rope",
    "dispatch_rope",
    "dispatched_conv2d",
    "dispatched_sliding_window_attention",
    "flash_sliding_window_tile",
    "get_sliding_window_bounds",
    "precompute_axial_freqs",
    "precompute_axial_freqs_cis",
    "precompute_freqs",
    "precompute_freqs_cis",
    "rotate_half",
    "sliding_window_attention",
]
