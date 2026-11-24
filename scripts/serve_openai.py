#!/usr/bin/env python3
"""Thin forwarding script for parallax.cli.serve_openai."""
from parallax.cli.serve_openai import main, run_server  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
