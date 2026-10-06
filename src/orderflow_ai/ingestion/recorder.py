"""Append-only tick recorder.

Recording is what makes the rest of the system reproducible: collected files
feed the replay feed, the backtester and the test-suite fixtures. Writes are
buffered and flushed periodically to keep the hot path free of disk I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, Any

from ..core.models import BookSnapshot, DepthUpdate, Trade
from .feeds.replay import RECORD_TYPE_DEPTH, RECORD_TYPE_TRADE

__all__ = ["TickRecorder"]

logger = logging.getLogger(__name__)


class TickRecorder:
    """Writes trades and book snapshots as JSON lines."""

    def __init__(self, path: str | Path, flush_every: int = 500, book_interval: float = 5.0) -> None:
        """Create a recorder.

        Args:
            path: Destination file. Parent directories are created on demand.
            flush_every: Flush the file after this many written records.
            book_interval: Minimum number of seconds between two book
                snapshots. Book state changes slowly next to trades, so
                throttling keeps files small.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = self.path.open("a", encoding="utf-8")
        self._flush_every = max(1, flush_every)
        self._written = 0
        self._book_interval = book_interval
        self._last_book_ts: float | None = None

    @property
    def written(self) -> int:
        """Number of records written so far."""
        return self._written

    @property
    def book_interval(self) -> float:
        """Minimum number of seconds between two recorded book snapshots."""
        return self._book_interval

    def record_trade(self, trade: Trade) -> None:
        """Append one trade record."""
        self._write(
            {
                "type": RECORD_TYPE_TRADE,
                "ts": round(trade.ts, 6),
                "symbol": trade.symbol,
                "price": trade.price,
                "size": trade.size,
                "side": trade.side.value,
                "source": trade.source,
            }
        )

    def record_book(self, book: BookSnapshot, ts: float, force: bool = False) -> None:
        """Append a book snapshot if the throttle window has elapsed."""
        if not book:
            return
        if not force and self._last_book_ts is not None and ts - self._last_book_ts < self._book_interval:
            return
        self._last_book_ts = ts
        self._write(
            {
                "type": RECORD_TYPE_DEPTH,
                "ts": round(ts, 6),
                "snapshot": True,
                "bids": [[level.price, level.size] for level in book.bids],
                "asks": [[level.price, level.size] for level in book.asks],
            }
        )

    def record_update(self, update: DepthUpdate) -> None:
        """Append a raw depth update."""
        self._write(
            {
                "type": RECORD_TYPE_DEPTH,
                "ts": round(update.ts, 6),
                "snapshot": update.is_snapshot,
                "bids": [[level.price, level.size] for level in update.bids],
                "asks": [[level.price, level.size] for level in update.asks],
            }
        )

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __enter__(self) -> TickRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _write(self, record: dict[str, Any]) -> None:
        """Serialise and buffer one record."""
        self._handle.write(json.dumps(record, separators=(",", ":")))
        self._handle.write("\n")
        self._written += 1
        if self._written % self._flush_every == 0:
            self._handle.flush()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown safety
        try:
            self.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("recorder close failed: %s", exc)