"""Stateful order book maintained from streamed depth updates."""

from __future__ import annotations

from orderflow_ai.core.models import BookSnapshot, DepthLevel, DepthUpdate, Side

__all__ = ["OrderBook"]


class OrderBook:
    """Applies depth diffs and exposes price-sorted snapshots.

    A ``DepthUpdate`` with ``is_snapshot=True`` replaces both sides. A diff only
    touches the levels it mentions and a size of zero removes a level, which
    matches how every exchange streams incremental book changes.
    """

    __slots__ = ("_bids", "_asks", "_depth_levels", "_last_update")

    def __init__(self, depth_levels: int = 20) -> None:
        """Create an empty book.

        Args:
            depth_levels: Maximum levels retained per side. Excess levels are
                discarded because nothing downstream reads past this depth.
        """
        if depth_levels <= 0:
            raise ValueError("depth_levels must be greater than zero")
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._depth_levels = depth_levels
        self._last_update: float = 0.0

    @property
    def depth_levels(self) -> int:
        """Maximum retained levels per side."""
        return self._depth_levels

    @property
    def last_update(self) -> float:
        """Timestamp of the most recently applied update."""
        return self._last_update

    def apply(self, update: DepthUpdate) -> BookSnapshot:
        """Apply ``update`` and return the resulting snapshot."""
        self._last_update = update.ts
        if update.is_snapshot:
            self._bids = {level.price: level.size for level in update.bids}
            self._asks = {level.price: level.size for level in update.asks}
        else:
            _merge(self._bids, update.bids)
            _merge(self._asks, update.asks)
        self._trim()
        return self.snapshot()

    def snapshot(self) -> BookSnapshot:
        """Return the current book as an immutable, sorted snapshot."""
        return BookSnapshot(
            bids=(DepthLevel(price, size) for price, size in sorted(self._bids.items(), reverse=True)),
            asks=(DepthLevel(price, size) for price, size in sorted(self._asks.items())),
        )

    def reset(self) -> None:
        """Clear both sides of the book."""
        self._bids.clear()
        self._asks.clear()

    def _trim(self) -> None:
        """Keep only the best ``depth_levels`` prices on each side."""
        if len(self._bids) > self._depth_levels:
            for price in sorted(self._bids)[self._depth_levels :]:
                del self._bids[price]
        if len(self._asks) > self._depth_levels:
            for price in sorted(self._asks, reverse=True)[self._depth_levels :]:
                del self._asks[price]


def _merge(side: dict[float, float], levels: tuple[DepthLevel, ...]) -> None:
    """Merge ``levels`` into ``side`` in place, dropping zero sized levels."""
    for level in levels:
        if level.size <= 0:
            side.pop(level.price, None)
        else:
            side[level.price] = level.size


def side_depth(snapshot: BookSnapshot, side: Side, levels: int = 5) -> float:
    """Total resting size of one side of ``snapshot``."""
    return snapshot.bid_depth(levels) if side is Side.BUY else snapshot.ask_depth(levels)