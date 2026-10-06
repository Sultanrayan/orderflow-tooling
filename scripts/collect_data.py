#!/usr/bin/env python
"""Record ticks from a live exchange feed for later replay.

Example::

    python scripts/collect_data.py --symbol BTCUSDT --duration 3600
"""

from __future__ import annotations

import sys

from bootstrap import ensure_importable

ensure_importable()

from orderflow_ai.cli import main  # noqa: E402 - import after path setup

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "collect" not in argv:
        argv = ["collect", *argv]
    raise SystemExit(main(argv))