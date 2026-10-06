#!/usr/bin/env python
"""Run the live order flow pipeline.

Examples::

    python scripts/live_run.py --config configs/binance.yaml
    python scripts/live_run.py --config configs/default.yaml --max-bars 5 --payload
"""

from __future__ import annotations

import sys

from bootstrap import ensure_importable

ensure_importable()

from orderflow_ai.cli import main  # noqa: E402 - import after path setup

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "live" not in argv:
        argv = ["live", *argv]
    raise SystemExit(main(argv))