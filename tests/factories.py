"""Helpers for building deterministic market data in tests.

Keeping the factories in one module makes every test file independent of the
others while still exercising the real data models rather than mocks.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable

from orderflow_ai.core.config import AIConfig, ExchangeConfig, OrderFlowConfig, PipelineConfig
from orderflow_ai.core.models import (
    BookSnapshot,
    Candle,
    DepthLevel,
    FootprintBar,
    FootprintLevel,
    Side,
    Trade,
)
from orderflow_ai.core.timeframes import Timeframe, bar_open_time

DEFAULT_START = 1_700_000_000.0
PRICE_DECIMALS = 2

__all__ = [
    "DEFAULT_START",
    "make_bar",
    "make_book",
    "make_candle",
    "make_config",
    "make_trade",
    "make_trades",
    "zigzag_candles",
]


def make_trade(
    price: float,
    size: float = 1.0,
    side: Side = Side.BUY,
    ts: float = DEFAULT_START,
    symbol: str = "BTCUSDT",
) -> Trade:
    """Create a single trade."""
    return Trade(symbol=symbol, ts=ts, price=price, size=size, side=side, source="test")


def make_trades(
    rows: Iterable[tuple[float, float, Side]],
    start_ts: float = DEFAULT_START,
    step: float = 1.0,
) -> list[Trade]:
    """Create trades from ``(price, size, side)`` tuples."""
    return [
        make_trade(price, size, side, ts=start_ts + index * step)
        for index, (price, size, side) in enumerate(rows)
    ]


def make_bar(
    timeframe: Timeframe = Timeframe.M1,
    open_ts: float = DEFAULT_START,
    levels: dict[float, tuple[float, float]] | None = None,
    trades_per_level: int = 1,
    closed: bool = True,
    symbol: str = "BTCUSDT",
    close: float | None = None,
) -> FootprintBar:
    """Create a footprint bar from ``{price: (bid_volume, ask_volume)}``.

    Args:
        timeframe: Bar duration.
        open_ts: Bar open timestamp in epoch seconds.
        levels: Volume pairs per price, split by aggressor side. Missing sides
            default to ``0``.
        trades_per_level: Executions recorded on each side of every level. This
            drives average clip size and iceberg style heuristics.
        closed: Whether the bar is marked closed.
        symbol: Symbol stored on the bar.
        close: Close price. Defaults to the highest traded price; pass it
            explicitly for patterns where price rejected back inside the range.
    """
    bar = FootprintBar(
        timeframe=timeframe,
        open_time=bar_open_time(open_ts, timeframe),
        open=0.0,
        high=0.0,
        low=0.0,
        close=0.0,
        symbol=symbol,
        closed=closed,
    )
    if not levels:
        return bar

    prices = sorted(levels)
    bar.open = prices[0]
    bar.high = prices[-1]
    bar.low = prices[0]
    bar.close = close if close is not None else prices[-1]

    for price, (bid_volume, ask_volume) in sorted(levels.items()):
        level = FootprintLevel(price=price, bid_volume=bid_volume, ask_volume=ask_volume)
        level.bid_trades = trades_per_level if bid_volume else 0
        level.ask_trades = trades_per_level if ask_volume else 0
        bar.levels[price] = level
    return bar


def make_book(
    mid: float = 100.0,
    bid_size: float = 10.0,
    ask_size: float = 10.0,
    levels: int = 5,
    step: float = 0.5,
) -> BookSnapshot:
    """Create a symmetric or skewed book around ``mid``.

    Args:
        mid: Mid price.
        bid_size: Size resting on each bid level.
        ask_size: Size resting on each ask level.
        levels: Number of levels per side.
        step: Price distance between levels.
    """
    bids = [DepthLevel(mid - step * index, bid_size) for index in range(1, levels + 1)]
    asks = [DepthLevel(mid + step * index, ask_size) for index in range(1, levels + 1)]
    return BookSnapshot(bids=bids, asks=asks)


def make_candle(
    index: int = 0,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    timeframe: Timeframe = Timeframe.M1,
) -> Candle:
    """Create one candle ``index`` minutes after the default start."""
    open_ts = DEFAULT_START + index * timeframe.seconds
    open_time = dt.datetime.fromtimestamp(open_ts, tz=dt.UTC)
    return Candle(
        open_time=open_time,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=1.0,
        close_time=open_time + dt.timedelta(seconds=timeframe.seconds),
    )


def zigzag_candles(
    count: int = 20,
    trend: float = 0.1,
    amplitude: float = 2.0,
    period: int = 8,
    start_price: float = 100.0,
    dampen_after: int | None = None,
    damped_amplitude: float = 0.8,
) -> list[Candle]:
    """Build a deterministic zigzag of candles with clean fractal pivots.

    A sine sampled over ``period`` bars puts its peaks and troughs on single
    bars, so every turning point becomes a confirmed swing. ``trend`` lifts each
    cycle, which produces higher highs and higher lows; ``dampen_after`` shrinks
    the amplitude instead, which produces a range with a change of character.

    Args:
        count: Number of candles.
        trend: Price added per bar.
        amplitude: Swing amplitude in price units.
        period: Bars per full cycle.
        start_price: Price of the first bar.
        dampen_after: Bar index from which ``damped_amplitude`` applies.
        damped_amplitude: Amplitude used after ``dampen_after``.

    Returns:
        The candles, oldest first.
    """
    candles: list[Candle] = []
    for index in range(count):
        swing = amplitude if dampen_after is None or index < dampen_after else damped_amplitude
        close = start_price + trend * index + swing * math.sin(2 * math.pi * index / period)
        candles.append(
            make_candle(
                index=index,
                open_=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
            )
        )
    return candles


def make_config(
    mode: str = "synthetic",
    timeframe: Timeframe = Timeframe.M1,
    history_bars: int = 20,
    price_precision: int = PRICE_DECIMALS,
    tick_size: float = 0.5,
    **ai_kwargs,
) -> OrderFlowConfig:
    """Build a configuration suited to tests.

    Args:
        mode: Pipeline mode.
        timeframe: Base candle timeframe.
        history_bars: Lookback depth.
        price_precision: Decimals used for price bucketing.
        tick_size: Smallest price increment.
        **ai_kwargs: Overrides for :class:`~orderflow_ai.core.config.AIConfig`.
    """
    return OrderFlowConfig(
        exchange=ExchangeConfig(
            symbol="BTCUSDT",
            tick_size=tick_size,
            price_precision=price_precision,
            depth_levels=10,
        ),
        pipeline=PipelineConfig(
            mode=mode,
            timeframe=timeframe,
            history_bars=history_bars,
            context_timeframes=("5m",),
        ),
        ai=AIConfig(**ai_kwargs),
    )