"""Deterministic synthetic trade feed.

The synthetic feed makes the whole pipeline runnable without network access:
tests, demos and CI all use it. It is a mean-reverting random walk with a
per trade aggressor bias, so the resulting bars contain genuine order flow
imbalance rather than pure noise.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, fields, replace

from ...core.config import ExchangeConfig
from ...core.models import BookSnapshot, DepthLevel, DepthUpdate, Side, Trade
from ...core.timeframes import Timeframe
from ..order_book import OrderBook
from .base import TradeFeed

__all__ = ["SyntheticFeed"]


@dataclass(slots=True)
class SyntheticFeedConfig:
    """Tunables of the synthetic generator."""

    start_price: float = 100.0
    tick_size: float = 0.01
    start_time: float = 1_700_000_000.0
    trades_per_bar: int = 40
    volatility: float = 0.6
    mean_reversion: float = 0.02
    buy_bias: float = 0.5
    base_size: float = 1.0
    size_spread: float = 1.0
    depth_levels: int = 20
    book_size: float = 25.0


class SyntheticFeed(TradeFeed):
    """Generates a realistic trade and book stream from a fixed seed."""

    name = "synthetic"

    def __init__(
        self,
        exchange: ExchangeConfig | None = None,
        timeframe: Timeframe = Timeframe.M1,
        bars: int = 60,
        seed: int = 7,
        **overrides: float | int,
    ) -> None:
        """Create the generator.

        Args:
            exchange: Exchange settings, used for symbol, tick size and depth.
            timeframe: Timeframe whose length defines one bar of trades.
            bars: Number of bars to generate before the stream ends.
            seed: Random seed. The same seed always produces the same session.
            **overrides: Any :class:`SyntheticFeedConfig` field, for example
                ``volatility=1.2``.
        """
        self.exchange = exchange or ExchangeConfig()
        self.timeframe = timeframe
        self.bars = bars
        defaults = SyntheticFeedConfig(
            tick_size=self.exchange.tick_size or 0.01,
            depth_levels=self.exchange.depth_levels,
        )
        self.settings = _with_overrides(defaults, overrides)
        self._random = random.Random(seed)
        self._book = OrderBook(self.settings.depth_levels)

    async def stream(self) -> AsyncIterator[Trade]:
        """Yield trades for ``bars`` consecutive bars, oldest first."""
        settings = self.settings
        price = settings.start_price
        ticks_per_bar = max(1, settings.trades_per_bar)
        step = settings.tick_size
        total_bars = max(1, self.bars)

        for bar_index in range(total_bars):
            bar_start = settings.start_time + bar_index * self.timeframe.seconds
            anchor = price
            for tick_index in range(ticks_per_bar):
                ts = bar_start + (tick_index * self.timeframe.seconds) / ticks_per_bar

                # Mean reversion towards the bar anchor plus a random kick.
                drift = settings.mean_reversion * (anchor - price) / max(anchor, 1e-9)
                price += drift + self._random.gauss(0.0, settings.volatility * step)
                price = max(step, round(price / step) * step)

                side = self._next_side(bar_index, tick_index)
                size = max(
                    step,
                    round(
                        (settings.base_size + abs(self._random.gauss(0, settings.size_spread)))
                        / step
                    )
                    * step,
                )
                yield Trade(
                    symbol=self.exchange.symbol.upper(),
                    ts=ts,
                    price=price,
                    size=size,
                    side=side,
                    source=self.name,
                )
            self._book.apply(self._build_book(price, bar_start + self.timeframe.seconds))

    def snapshot(self) -> BookSnapshot:
        """Return the most recently generated book."""
        return self._book.snapshot()

    def _next_side(self, bar_index: int, tick_index: int) -> Side:
        """Choose the aggressor, occasionally producing a dominant run."""
        settings = self.settings
        if self._random.random() < 0.06:
            # Occasional one sided burst: aggressive flow into the book.
            return Side.BUY if self._random.random() < 0.5 else Side.SELL
        bias = settings.buy_bias
        if bar_index % 7 == 0:
            bias = min(0.9, bias + 0.2)
        elif bar_index % 11 == 0:
            bias = max(0.1, bias - 0.2)
        if tick_index % 23 == 0:
            bias = 1.0 - bias
        return Side.BUY if self._random.random() < bias else Side.SELL

    def _build_book(self, mid: float, ts: float) -> DepthUpdate:
        """Build a symmetric book around ``mid`` with a random size skew."""
        settings = self.settings
        step = settings.tick_size
        skew = self._random.uniform(0.7, 1.3)
        bids = [
            DepthLevel(round(mid - step * index, 6), settings.book_size * skew + self._random.random() * 10)
            for index in range(1, settings.depth_levels + 1)
        ]
        asks = [
            DepthLevel(round(mid + step * index, 6), settings.book_size / skew + self._random.random() * 10)
            for index in range(1, settings.depth_levels + 1)
        ]
        return DepthUpdate(ts=ts, bids=tuple(bids), asks=tuple(asks), is_snapshot=True)


def _with_overrides(
    config: SyntheticFeedConfig, overrides: dict[str, float | int]
) -> SyntheticFeedConfig:
    """Return a copy of ``config`` with ``overrides`` applied.

    Raises:
        ValueError: If an override does not name a known setting.
    """
    known = {field.name for field in fields(config)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown synthetic feed options: {', '.join(sorted(unknown))}")
    return replace(config, **overrides)