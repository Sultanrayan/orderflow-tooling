"""Value objects for market context."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..core.timeframes import Timeframe

__all__ = [
    "KeyLevels",
    "MarketContext",
    "MarketStructure",
    "MultiTimeframe",
    "SessionContext",
    "StructureLabel",
    "TimeframeState",
    "VolatilityState",
]

# Market structure vocabulary.
STRUCTURE_UPTREND = "uptrend"
STRUCTURE_DOWNTREND = "downtrend"
STRUCTURE_RANGE = "range"
STRUCTURE_UNKNOWN = "unknown"

# Break of structure vocabulary.
BOS_BULLISH = "bullish"
BOS_BEARISH = "bearish"
BOS_NONE = "none"

# Swing label vocabulary.
SWING_HH = "HH"
SWING_HL = "HL"
SWING_LH = "LH"
SWING_LL = "LL"

# Timeframe vocabulary.
ALIGNMENT_ALIGNED = "aligned"
ALIGNMENT_MIXED = "mixed"
ALIGNMENT_CONFLICT = "conflict"

# Volatility vocabulary.
VOLATILITY_LOW = "low"
VOLATILITY_NORMAL = "normal"
VOLATILITY_HIGH = "high"
VOLATILITY_EXTREME = "extreme"

StructureLabel = str


@dataclass(frozen=True, slots=True)
class MarketStructure:
    """Swing structure, trend label and the latest break of structure."""

    label: StructureLabel
    bos: str
    choch: bool = False
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    swing_highs: tuple[float, ...] = ()
    swing_lows: tuple[float, ...] = ()

    @property
    def is_bullish(self) -> bool:
        """``True`` when the structure is an uptrend."""
        return self.label == STRUCTURE_UPTREND

    @property
    def is_bearish(self) -> bool:
        """``True`` when the structure is a downtrend."""
        return self.label == STRUCTURE_DOWNTREND


@dataclass(frozen=True, slots=True)
class KeyLevels:
    """Support and resistance prices, nearest first."""

    support: tuple[float, ...] = ()
    resistance: tuple[float, ...] = ()
    poc: float | None = None
    value_area_high: float | None = None
    value_area_low: float | None = None

    def nearest_support(self, price: float) -> float | None:
        """Closest support at or below ``price``."""
        candidates = [level for level in self.support if level <= price]
        return max(candidates) if candidates else None

    def nearest_resistance(self, price: float) -> float | None:
        """Closest resistance at or above ``price``."""
        candidates = [level for level in self.resistance if level >= price]
        return min(candidates) if candidates else None


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Trading session the analysed bar belongs to."""

    primary: str
    active: tuple[str, ...] = ()
    is_london_open: bool = False
    is_new_york_open: bool = False
    minutes_into_session: float = 0.0


@dataclass(frozen=True, slots=True)
class TimeframeState:
    """One higher timeframe in the alignment calculation."""

    timeframe: Timeframe
    direction: str
    change_pct: float
    bars: int


@dataclass(frozen=True, slots=True)
class MultiTimeframe:
    """Higher timeframe agreement."""

    direction: str
    score: float
    states: tuple[TimeframeState, ...] = field(default_factory=tuple)

    @property
    def aligned(self) -> bool:
        """``True`` when every inspected timeframe agrees."""
        return self.direction == ALIGNMENT_ALIGNED


@dataclass(frozen=True, slots=True)
class VolatilityState:
    """Current volatility regime and its inputs."""

    regime: str
    average_range: float
    percentile: float
    atr_ratio: float = 1.0

    @property
    def is_expanded(self) -> bool:
        """``True`` for the ``high`` and ``extreme`` regimes."""
        return self.regime in {VOLATILITY_HIGH, VOLATILITY_EXTREME}


@dataclass(frozen=True, slots=True)
class MarketContext:
    """Everything layer 5 needs to describe the market around the signal."""

    symbol: str
    timeframe: Timeframe
    open_time: dt.datetime
    price: float
    structure: MarketStructure
    levels: KeyLevels
    session: SessionContext
    multi_timeframe: MultiTimeframe
    volatility: VolatilityState

    @property
    def level_proximity(self) -> float:
        """Distance from ``price`` to the closest key level as a ``0``-``1`` score.

        ``1.0`` means price is sitting exactly on a level, ``0.0`` means the
        closest level is more than one average bar range away.
        """
        nearest = self.nearest_level_price
        if nearest is None:
            return 0.0
        span = self.volatility.average_range or 1e-9
        return max(0.0, 1.0 - abs(self.price - nearest) / span)

    @property
    def nearest_level_price(self) -> float | None:
        """Closest support or resistance to the current price."""
        candidates = [
            level
            for level in (*self.levels.support, *self.levels.resistance)
            if level > 0
        ]
        return min(candidates, key=lambda level: abs(level - self.price)) if candidates else None