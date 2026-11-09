"""
Continuous Batching and Iteration-Level Request Scheduler.

Implements high-throughput multi-request dynamic batching with per-sequence
KV cache tracking and token-by-token generation for ParallaxLM without vLLM.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from parallax.generator import sample_next_token
from parallax.models.gpt import ParallaxLM


class RequestStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


@dataclass
class SequenceRequest:
    request_id: str
    prompt_tokens: List[int]
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_token_id: Optional[int] = None

    status: RequestStatus = RequestStatus.WAITING
    generated_tokens: List[int] = field(default_factory=list)
    # KV cache per layer: List of (K, V) ndarrays of shape [1, num_kv_heads, past_len, head_dim]
    kv_cache: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    total_tokens_seen: int = 0
