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
    FAILED = "FAILED"


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
    error: Optional[str] = None


class ContinuousBatchEngine:
    """Continuous batching serving engine for ParallaxLM.

    Dynamically admits requests up to max_batch_size, runs iteration-level
    decoding with per-sequence KV state preservation, and ejects completed sequences.
    """

    def __init__(self, model: ParallaxLM, max_batch_size: int = 8):
        self.model = model
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[SequenceRequest] = []
        self.running_requests: List[SequenceRequest] = []
        self.completed_requests: Dict[str, SequenceRequest] = {}

    def add_request(self, request: SequenceRequest) -> None:
        """Enqueues a new inference request."""
        self.waiting_queue.append(request)

    def has_unfinished_requests(self) -> bool:
        """Returns True if any requests are queued or currently decoding."""
        return bool(self.waiting_queue or self.running_requests)

    def step(self) -> List[Tuple[str, int]]:
        """Executes one iteration step across active sequences in the continuous batch.

        Returns:
            List of (request_id, newly_generated_token_id) for this step.
        """
        # Admit new requests up to max_batch_size
        while self.waiting_queue and len(self.running_requests) < self.max_batch_size:
            req = self.waiting_queue.pop(0)
            req.status = RequestStatus.RUNNING
            self.running_requests.append(req)

        if not self.running_requests:
            return []

        step_outputs: List[Tuple[str, int]] = []
        still_running: List[SequenceRequest] = []

        # Process each running request for 1 iteration
        for req in self.running_requests:
            try:
                if req.kv_cache is None:
                    # Prefill step
                    input_tokens = np.array([req.prompt_tokens], dtype=np.int64)
                    # Initialize empty KV cache per layer
                    num_heads = self.model.config.num_heads
                    num_kv_heads = (
                        self.model.config.num_kv_heads
                        if self.model.config.num_kv_heads is not None
                        else num_heads
                    )
                    head_dim = self.model.config.embed_dim // num_heads
                    initial_kv = [
                        (
                            np.empty((1, num_kv_heads, 0, head_dim), dtype=np.float64),
                            np.empty((1, num_kv_heads, 0, head_dim), dtype=np.float64),
                        )
                        for _ in range(self.model.config.num_layers)
                    ]
                    logits, updated_kv = self.model(input_tokens, kv_caches=initial_kv)
                    req.kv_cache = updated_kv
                    req.total_tokens_seen = len(req.prompt_tokens)
                    next_token_logits = logits.data[0, -1, :]
                else:
                    # Decode single token step
                    last_token = req.generated_tokens[-1]
                    input_tokens = np.array([[last_token]], dtype=np.int64)
                    logits, updated_kv = self.model(input_tokens, kv_caches=req.kv_cache)
                    req.kv_cache = updated_kv
                    req.total_tokens_seen += 1
                    next_token_logits = logits.data[0, -1, :]

                # Sample next token
                next_token = sample_next_token(
                    next_token_logits,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                )

                req.generated_tokens.append(next_token)
                step_outputs.append((req.request_id, next_token))

                # Check completion criteria
                reached_max = len(req.generated_tokens) >= req.max_new_tokens
                hit_eos = req.eos_token_id is not None and next_token == req.eos_token_id

                if reached_max or hit_eos:
                    req.status = RequestStatus.FINISHED
                    self.completed_requests[req.request_id] = req
                else:
                    still_running.append(req)
            except Exception as e:
                req.status = RequestStatus.FAILED
                req.error = str(e)
                self.completed_requests[req.request_id] = req

        self.running_requests = still_running
        return step_outputs

