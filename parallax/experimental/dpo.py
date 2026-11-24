"""Direct Preference Optimization (DPO) experimental module."""

from __future__ import annotations

from parallax.rl import compute_sequence_log_probs, dpo_loss

__all__ = ["compute_sequence_log_probs", "dpo_loss"]
