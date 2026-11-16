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

__all__ = [
    "AxialRoPEFrequencyTable",
    "DispatchedConv2d",
    "KVCacheBuffer",
    "KVCacheManager",
    "RoPEFrequencyTable",
    "apply_axial_rope",
    "apply_axial_rope_grid",
    "apply_rotary_emb",
    "conv2d_backward",
    "conv2d_forward",
    "create_kv_cache",
    "dispatch_axial_rope",
    "dispatch_rope",
    "dispatched_conv2d",
    "precompute_axial_freqs",
    "precompute_axial_freqs_cis",
    "precompute_freqs",
    "precompute_freqs_cis",
    "rotate_half",
]
