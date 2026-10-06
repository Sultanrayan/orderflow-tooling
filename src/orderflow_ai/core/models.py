"""Canonical market data models shared by every layer of the engine.

Layer 1 produces these objects, layers 2 to 4 consume them and layer 5 maps
them onto a token optimised JSON payload. Keeping a single canonical shape means
the rest of the codebase never has to know which exchange a tick came from.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .timeframes import Timeframe

__all__ = [
    "Side",
    "Trade",
    "DepthLevel",
    "DepthUpdate",
    "BookSnapshot",
    "Candle",
    "FootprintLevel",
    "FootprintBar",
]


class Side(str, Enum):
    """Aggressor side of a trade or side of the order book."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        """The other side."""
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> float:
        """``+1`` for buys and ``-1`` for sells, handy for signed arithmetic."""
        return 1.0 if self is Side.BUY else -1.0


@dataclass(slots=True, frozen=True)
class Trade:
    """A single executed trade, normalised across exchanges.

    Attributes:
        symbol: Trading symbol, for example ``BTCUSDT``.
        ts: Exchange event time in epoch seconds.
        price: Execution price.
        size: Executed base quantity.
        side: Aggressing side. A buyer lifting the offer is ``Side.BUY``.
        source: Exchange or feed identifier, used for diagnostics.
    """

    symbol: str
    ts: float
    price: float
    size: float
    side: Side
    source: str = "unknown"

    @property
    def signed_size(self) -> float:
        """Size signed by aggressor, positive for buys and negative for sells."""
        return self.size * self.side.sign


@dataclass(slots=True, frozen=True)
class DepthLevel:
    """One price level of the order book."""

    price: float
    size: float


@dataclass(slots=True, frozen=True)
class DepthUpdate:
    """An order book update.

    Either a full ``is_snapshot`` replacement of both sides, or a diff where the
    tuples contain only the levels that changed. A size of ``0`` in a diff means
    "remove this level".
    """

    ts: float
    bids: tuple[DepthLevel, ...] = ()
    asks: tuple[DepthLevel, ...] = ()
    is_snapshot: bool = False


class BookSnapshot:
    """Immutable, price-sorted view of one order book state."""

    __slots__ = ("bids", "asks")

    def __init__(
        self,
        bids: Iterable[DepthLevel] = (),
        asks: Iterable[DepthLevel] = (),
    ) -> None:
        self.bids: tuple[DepthLevel, ...] = tuple(sorted(bids, key=_price_desc))
        self.asks: tuple[DepthLevel, ...] = tuple(sorted(asks, key=_price_asc))

    @classmethod
    def empty(cls) -> BookSnapshot:
        """Return a snapshot with no levels on either side."""
        return cls()

    @property
    def best_bid(self) -> float | None:
        """Highest bid price, or ``None`` when the bid side is empty."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Lowest ask price, or ``None`` when the ask side is empty."""
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        """Mid price, or ``None`` when either side is empty."""
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2

    @property
    def spread(self) -> float | None:
        """Absolute spread between best bid and best ask."""
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price

    @property
    def spread_bps(self) -> float | None:
        """Spread expressed in basis points of the mid price."""
        spread, mid = self.spread, self.mid
        if spread is None or not mid:
            return None
        return spread / mid * 10_000

    def bid_depth(self, levels: int) -> float:
        """Total resting size within the top ``levels`` bid levels."""
        return sum(level.size for level in self.bids[:levels])

    def ask_depth(self, levels: int) -> float:
        """Total resting size within the top ``levels`` ask levels."""
        return sum(level.size for level in self.asks[:levels])

    def side(self, side: Side) -> tuple[DepthLevel, ...]:
        """Return the bid or ask ladder of this snapshot."""
        return self.bids if side is Side.BUY else self.asks

    def depth_ratio(self, levels: int = 5) -> float | None:
        """Bid over ask size ratio within ``levels``, or ``None`` if unavailable."""
        ask = self.ask_depth(levels)
        if ask <= 0:
            return None
        return self.bid_depth(levels) / ask

    def __bool__(self) -> bool:
        return bool(self.bids or self.asks)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"BookSnapshot(bids={len(self.bids)}, asks={len(self.asks)})"


@dataclass(slots=True, frozen=True)
class Candle:
    """An OHLCV candle."""

    open_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: dt.datetime
    trade_count: int = 0
    closed: bool = True

    @property
    def range(self) -> float:
        """High minus low, the raw volatility of the bar."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Signed close minus open."""
        return self.close - self.open

    @property
    def mid(self) -> float:
        """Mid point of the bar range."""
        return (self.high + self.low) / 2

    @property
    def is_up(self) -> bool:
        """``True`` when the bar closed above its open."""
        return self.close >= self.open


@dataclass(slots=True)
class FootprintLevel:
    """Bid and ask volume traded at one price inside a bar.

    Volume is split by aggressor: aggressive buys lift the offer and therefore
    add to ``ask_volume``, aggressive sells hit the bid and add to
    ``bid_volume``. Trade counts support iceberg style heuristics.
    """

    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    bid_trades: int = 0
    ask_trades: int = 0

    @property
    def total_volume(self) -> float:
        """Total volume traded at this level."""
        return self.bid_volume + self.ask_volume

    @property
    def delta(self) -> float:
        """Buy minus sell volume at this level."""
        return self.ask_volume - self.bid_volume

    @property
    def trade_count(self) -> int:
        """Number of executions at this level."""
        return self.bid_trades + self.ask_trades

    @property
    def average_trade_size(self) -> float:
        """Mean execution size at this level, ``0.0`` when there were no fills."""
        return self.total_volume / self.trade_count if self.trade_count else 0.0

    @property
    def imbalance_ratio(self) -> float:
        """Larger side divided by smaller side. ``1.0`` means perfectly balanced."""
        if self.bid_volume <= 0 or self.ask_volume <= 0:
            return float("inf") if self.total_volume > 0 else 1.0
        return max(self.bid_volume, self.ask_volume) / min(self.bid_volume, self.ask_volume)

    @property
    def imbalance_side(self) -> Side | None:
        """Side that dominates this level, or ``None`` when it is balanced."""
        if self.imbalance_ratio <= 1.0:
            return None
        return Side.BUY if self.ask_volume > self.bid_volume else Side.SELL


@dataclass(slots=True)
class FootprintBar:
    """A closed or in-progress bar enriched with per-price volume.

    This is the primary input of layer 2. ``levels`` is kept sorted by price.
    """

    timeframe: Timeframe
    open_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    symbol: str = ""
    levels: dict[float, FootprintLevel] = field(default_factory=dict)
    closed: bool = False
    last_ts: float = 0.0

    def record(self, trade: Trade, price_decimals: int = 6) -> FootprintLevel:
        """Fold a trade into the bar and return the affected level.

        Args:
            trade: The executed trade to fold in.
            price_decimals: Decimal places used to bucket the price level.

        Returns:
            The footprint level the trade was recorded on.
        """
        price = round(trade.price, price_decimals)
        is_first_trade = not self.levels
        level = self.levels.get(price)
        if level is None:
            level = FootprintLevel(price=price)
            self.levels[price] = level

        if trade.side is Side.BUY:
            level.ask_volume += trade.size
            level.ask_trades += 1
        else:
            level.bid_volume += trade.size
            level.bid_trades += 1

        self.last_ts = max(self.last_ts, trade.ts)
        if is_first_trade:
            self.open = trade.price
        self.high = max(self.high, trade.price)
        self.low = min(self.low, trade.price)
        self.close = trade.price
        return level

    @property
    def sorted_levels(self) -> list[FootprintLevel]:
        """Levels ordered by price, low to high."""
        return [self.levels[price] for price in sorted(self.levels)]

    @property
    def total_volume(self) -> float:
        """Sum of all traded volume in the bar."""
        return sum(level.total_volume for level in self.levels.values())

    @property
    def buy_volume(self) -> float:
        """Aggressive buy volume in the bar."""
        return sum(level.ask_volume for level in self.levels.values())

    @property
    def sell_volume(self) -> float:
        """Aggressive sell volume in the bar."""
        return sum(level.bid_volume for level in self.levels.values())

    @property
    def delta(self) -> float:
        """Buy volume minus sell volume for the bar."""
        return self.buy_volume - self.sell_volume

    @property
    def trade_count(self) -> int:
        """Total executions recorded in the bar."""
        return sum(level.trade_count for level in self.levels.values())

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low

    @property
    def close_time(self) -> dt.datetime:
        """The instant at which this bar is considered closed."""
        return self.open_time + dt.timedelta(seconds=self.timeframe.seconds)

    def level_at(self, price: float, price_decimals: int = 6) -> FootprintLevel | None:
        """Return the level matching ``price``, or ``None`` when untouched."""
        return self.levels.get(round(price, price_decimals))

    def average_trade_size(self) -> float:
        """Mean execution size across the whole bar."""
        trades = self.trade_count
        return self.total_volume / trades if trades else 0.0

    def to_candle(self) -> Candle:
        """Project the bar onto a plain OHLCV candle."""
        return Candle(
            open_time=self.open_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.total_volume,
            close_time=self.close_time,
            trade_count=self.trade_count,
            closed=self.closed,
        )

    def __bool__(self) -> bool:
        return bool(self.levels)

    def __iter__(self) -> Iterator[FootprintLevel]:
        return iter(self.sorted_levels)


def _price_desc(level: DepthLevel) -> float:
    return -level.price


def _price_asc(level: DepthLevel) -> float:
    return level.price


def bars_to_candles(bars: Sequence[FootprintBar]) -> list[Candle]:
    """Convert a sequence of footprint bars into plain candles."""
    return [bar.to_candle() for bar in bars]