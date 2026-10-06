#!/usr/bin/env python
"""Paper trade a recorded session.

Examples::

    python scripts/backtest.py --start 2024-01-01 --end 2024-10-01
    python scripts/backtest.py --config configs/replay.yaml --file data/ticks.jsonl
"""

from __future__ import annotations

import sys

from bootstrap import ensure_importable

ensure_importable()

from orderflow_ai.cli import main  # noqa: E402 - import after path setup

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "backtest" not in argv:
        argv = ["backtest", *argv]
    raise SystemExit(main(argv))