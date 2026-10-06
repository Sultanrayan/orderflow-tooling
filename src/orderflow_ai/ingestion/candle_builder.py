"""Aggregation of raw ticks into footprint bars.

Layer 2 needs per-price volume, which only exists inside a bar, so layer 1 owns
the bucketing: trades are folded into a :class:`~orderflow_ai.core.models.FootprintBar`
and the previous bar is emitted as soon as a trade crosses into the next one.
"""

from __future__ import annotations

from ..core.models import FootprintBar, Trade
from ..core.timeframes import Timeframe, bar_open_time, floor_timestamp

__all__ = ["CandleBuilder"]


class CandleBuilder:
    """Turns a stream of trades into a stream of closed footprint bars."""

    __slots__ = ("_timeframe", "_price_decimals", "_bar", "_last_closed")

    def __init__(
        self,
        timeframe: Timeframe = Timeframe.M1,
        price_decimals: int = 6,
    ) -> None:
        """Create a builder for bars of ``timeframe`` length.

        Args:
            timeframe: Length of the produced bars.
            price_decimals: Decimal places used to bucket price levels.
        """
        self._timeframe = timeframe
        self._price_decimals = price_decimals
        self._bar: FootprintBar | None = None
        self._last_closed: FootprintBar | None = None

    @property
    def timeframe(self) -> Timeframe:
        """Timeframe of the produced bars."""
        return self._timeframe

    @property
    def current(self) -> FootprintBar | None:
        """The in-progress bar, or ``None`` before the first trade."""
        return self._bar

    @property
    def last_closed(self) -> FootprintBar | None:
        """The most recently completed bar."""
        return self._last_closed

    def push(self, trade: Trade) -> FootprintBar | None:
        """Add ``trade`` to the current bar.

        Args:
            trade: The trade to fold in.

        Returns:
            The bar that was completed by this trade, or ``None`` when the
            trade stayed inside the current bar.
        """
        open_ts = floor_timestamp(trade.ts, self._timeframe)
        if self._bar is None or open_ts != self._bar.open_time.timestamp():
            closed = self.flush()
            self._bar = _new_bar(self._timeframe, trade, open_ts, self._price_decimals)
            self._bar.record(trade, self._price_decimals)
            return closed

        self._bar.record(trade, self._price_decimals)
        return None

    def flush(self) -> FootprintBar | None:
        """Close the in-progress bar and return it.

        Returns:
            The closed bar, or ``None`` when no trade has been seen yet. A bar
            without volume is discarded because it would only add noise.
        """
        bar, self._bar = self._bar, None
        if bar is None or not bar.levels:
            return None
        bar.closed = True
        self._last_closed = bar
        return bar


def _new_bar(
    timeframe: Timeframe, trade: Trade, open_ts: float, price_decimals: int
) -> FootprintBar:
    """Create a bar seeded with the opening trade price."""
    price = round(trade.price, price_decimals)
    return FootprintBar(
        timeframe=timeframe,
        open_time=bar_open_time(open_ts, timeframe),
        open=price,
        high=price,
        low=price,
        close=price,
        symbol=trade.symbol,
    )