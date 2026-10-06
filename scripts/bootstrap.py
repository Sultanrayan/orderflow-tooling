"""Make the ``src`` layout importable when the package is not installed.

Running ``python scripts/live_run.py`` from a fresh clone should work without
``pip install -e .`` first. Importing this module and calling
:func:`ensure_importable` adds ``src`` to ``sys.path`` when the installed
package is not found.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["SRC_DIR", "ensure_importable"]

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def ensure_importable() -> None:
    """Add the repository ``src`` directory to ``sys.path`` if needed."""
    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


ensure_importable()