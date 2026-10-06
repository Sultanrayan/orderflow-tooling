"""Replay of recorded ticks from a JSON lines file.

``scripts/collect_data.py`` writes one JSON object per line. Trade records carry
``price``/``size``/``side``; book records carry ``bids``/``asks``. Replaying the
same file therefore reproduces an earlier session exactly, which is what the
backtester and the tests consume.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from ...core.models import BookSnapshot, DepthLevel, DepthUpdate, Side, Trade
from ..order_book import OrderBook
from .base import TradeFeed

__all__ = ["ReplayFeed", "read_records"]

RECORD_TYPE_TRADE = "trade"
RECORD_TYPE_DEPTH = "depth"


class ReplayFeed(TradeFeed):
    """Replays a recorded tick file as a trade feed with book state."""

    name = "replay"

    def __init__(
        self,
        path: str | Path,
        symbol: str = "BTCUSDT",
        price_decimals: int = 6,
        depth_levels: int = 20,
        speed: float = 0.0,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        """Create a replay feed.

        Args:
            path: JSON lines file produced by :class:`TickRecorder`.
            symbol: Symbol assigned to records that do not carry one.
            price_decimals: Decimal places used to bucket book prices.
            depth_levels: Book levels retained per side.
            speed: Artificial delay between records in seconds. ``0`` replays
                as fast as possible.
            start: Inclusive lower bound of the replay window, in epoch seconds.
            end: Exclusive upper bound of the replay window, in epoch seconds.
        """
        self.path = Path(path)
        self._symbol = symbol.upper()
        self._price_decimals = price_decimals
        self._speed = speed
        self._window = (start, end)
        self._book = OrderBook(depth_levels)

    async def stream(self) -> AsyncIterator[Trade]:
        """Yield the recorded trades in file order."""
        for record in read_records(self.path):
            if not self._in_window(record):
                continue
            if record.get("type") == RECORD_TYPE_DEPTH:
                self._book.apply(_to_depth_update(record, self._price_decimals))
                continue
            trade = _to_trade(record, self._symbol)
            if trade is None:
                continue
            if self._speed:
                await asyncio.sleep(self._speed)
            yield trade

    def _in_window(self, record: dict[str, Any]) -> bool:
        """``True`` when the record falls inside the configured window."""
        start, end = self._window
        if start is None and end is None:
            return True
        timestamp = float(record.get("ts") or 0.0)
        if start is not None and timestamp < start:
            return False
        return not (end is not None and timestamp >= end)

    def snapshot(self) -> BookSnapshot:
        """Return the book state built from the replayed depth records."""
        return self._book.snapshot()


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every JSON object stored in ``path``, skipping malformed lines."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Replay file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _to_trade(record: dict[str, Any], default_symbol: str) -> Trade | None:
    """Convert a recorded trade line into a :class:`Trade`."""
    price = float(record.get("price") or 0.0)
    size = float(record.get("size") or 0.0)
    if price <= 0 or size <= 0:
        return None
    side = Side.BUY if str(record.get("side", "buy")).lower() == "buy" else Side.SELL
    return Trade(
        symbol=str(record.get("symbol") or default_symbol).upper(),
        ts=float(record.get("ts") or 0.0),
        price=price,
        size=size,
        side=side,
        source=str(record.get("source") or "replay"),
    )


def _to_depth_update(record: dict[str, Any], price_decimals: int) -> DepthUpdate:
    """Convert a recorded book line into a :class:`DepthUpdate`."""
    return DepthUpdate(
        ts=float(record.get("ts") or 0.0),
        bids=tuple(_levels(record.get("bids"), price_decimals)),
        asks=tuple(_levels(record.get("asks"), price_decimals)),
        is_snapshot=bool(record.get("snapshot", True)),
    )


def _levels(rows: Any, price_decimals: int) -> Iterator[DepthLevel]:
    """Convert ``[[price, size], ...]`` rows into :class:`DepthLevel` items."""
    for row in rows or ():
        with contextlib.suppress(TypeError, ValueError, IndexError):
            yield DepthLevel(price=round(float(row[0]), price_decimals), size=float(row[1]))