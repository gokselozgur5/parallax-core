"""Compute backend registry."""

from parallax.backends.registry import clear, get, list_backends, register, unregister
from parallax.backends.report import generate_markdown_report

__all__ = ["clear", "generate_markdown_report", "get", "list_backends", "register", "unregister"]
