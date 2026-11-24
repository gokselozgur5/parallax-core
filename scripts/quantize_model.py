#!/usr/bin/env python3
"""Thin forwarding script for parallax.cli.quantize_model."""
from parallax.cli.quantize_model import main, quantize_model  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
