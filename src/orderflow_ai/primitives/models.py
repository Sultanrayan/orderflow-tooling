"""Value objects produced by layer 2.

The names and fields mirror the documented primitive output so that the shape is
easy to map between the JSON in the specification and the Python API.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..core.config import PrimitivesConfig
from ..core.models import BookSnapshot, FootprintBar, FootprintLevel, Side
from ..core.timeframes import Timeframe

__all__ = [
    "AbsorptionScore",
    "Cvd",
    "Delta",
    "DeltaDivergence",
    "DirectionPrimitives",
    "Footprint",
    "FootprintRow",
    "Imbalance",
    "ImbalanceLevel",
    "OrderBookDepth",
    "PressurePrimitives",
    "PrimitiveInput",
    "Primitives",
    "QualityPrimitives",
    "StructurePrimitives",
    "ValueMigration",
    "VolumeIntensity",
    "VolumeProfile",
    "Wall",
]

BIAS_BULLISH = "bullish"
BIAS_BEARISH = "bearish"
BIAS_NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class FootprintRow:
    """One price level of a footprint bar."""

    price: float
    bid_volume: float
    ask_volume: float

    @property
    def total_volume(self) -> float:
        """Volume traded at this level."""
        return self.bid_volume + self.ask_volume

    @property
    def delta(self) -> float:
        """Buy minus sell volume at this level."""
        return self.ask_volume - self.bid_volume

    @classmethod
    def of(cls, level: FootprintLevel) -> FootprintRow:
        """Build a row from a :class:`~orderflow_ai.core.models.FootprintLevel`."""
        return cls(price=level.price, bid_volume=level.bid_volume, ask_volume=level.ask_volume)


@dataclass(frozen=True, slots=True)
class Footprint:
    """Structure primitive: price level detail of the analysed bar."""

    open: float
    high: float
    low: float
    close: float
    total_volume: float
    buy_volume: float
    sell_volume: float
    delta: float
    trade_count: int
    levels: tuple[FootprintRow, ...] = ()

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    """Structure primitive: where volume was accepted over the lookback."""

    poc: float
    vah: float
    val: float
    hvn: tuple[float, ...] = ()
    lvn: tuple[float, ...] = ()
    total_volume: float = 0.0

    @property
    def value_area_width(self) -> float:
        """Distance between value area high and low."""
        return self.vah - self.val


@dataclass(frozen=True, slots=True)
class ValueMigration:
    """Structure primitive: direction and strength of the point of control move."""

    poc_direction: str
    migration_strength: float
    poc_current: float
    poc_previous: float
    ticks: float


@dataclass(frozen=True, slots=True)
class Delta:
    """Direction primitive: aggressive volume balance of the latest bar."""

    value: float
    ratio: float
    direction: str
    strength: float
    buy_volume: float
    sell_volume: float
    total_volume: float


@dataclass(frozen=True, slots=True)
class Cvd:
    """Direction primitive: cumulative volume delta and its trend."""

    value: float
    slope: float
    momentum: str
    bars: int


@dataclass(frozen=True, slots=True)
class DeltaDivergence:
    """Direction primitive: price and delta disagreeing."""

    detected: bool
    type: str | None = None
    confidence: float = 0.0
    bars_observed: int = 0
    price_change: float = 0.0
    delta_change: float = 0.0


@dataclass(frozen=True, slots=True)
class ImbalanceLevel:
    """A single price level whose two sided volume is lopsided."""

    price: float
    ratio: float
    side: Side
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    def to_row(self) -> tuple[float, str, float]:
        """Compact ``(price, side, ratio)`` encoding used by the payload."""
        return (self.price, self.side.value, self.ratio)


@dataclass(frozen=True, slots=True)
class Imbalance:
    """Pressure primitive: diagonal imbalances of the latest bar."""

    threshold: float
    levels: tuple[ImbalanceLevel, ...] = ()
    strongest: ImbalanceLevel | None = None

    @property
    def detected(self) -> bool:
        """``True`` when at least one level exceeds the configured threshold."""
        return bool(self.levels)

    @property
    def side(self) -> Side | None:
        """Side of the strongest imbalance, if any."""
        return self.strongest.side if self.strongest else None


@dataclass(frozen=True, slots=True)
class Wall:
    """A resting order that is large relative to its neighbours."""

    price: float
    side: Side
    size: float
    strength: float


@dataclass(frozen=True, slots=True)
class OrderBookDepth:
    """Pressure primitive: live book metrics."""

    bid_depth: float
    ask_depth: float
    imbalance_ratio: float
    imbalance_direction: str
    spread_bps: float | None = None
    walls: tuple[Wall, ...] = ()

    @property
    def total_depth(self) -> float:
        """Combined resting size on both sides."""
        return self.bid_depth + self.ask_depth


@dataclass(frozen=True, slots=True)
class VolumeIntensity:
    """Quality primitive: how heavy the bar is against its own baseline."""

    ratio: float
    percentile: float
    intensity: str
    volume: float
    average_volume: float


@dataclass(frozen=True, slots=True)
class AbsorptionScore:
    """Quality primitive: volume absorbed without price progress."""

    detected: bool
    confidence: float
    side: Side | None = None
    price: float | None = None
    volume_ratio: float = 0.0
    range_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class StructurePrimitives:
    """Footprint, volume profile and value migration."""

    footprint: Footprint
    volume_profile: VolumeProfile
    value_migration: ValueMigration


@dataclass(frozen=True, slots=True)
class DirectionPrimitives:
    """Delta, cumulative delta and delta divergence."""

    delta: Delta
    cvd: Cvd
    delta_divergence: DeltaDivergence


@dataclass(frozen=True, slots=True)
class PressurePrimitives:
    """Imbalance and live order book depth."""

    imbalance: Imbalance
    orderbook: OrderBookDepth


@dataclass(frozen=True, slots=True)
class QualityPrimitives:
    """Volume intensity and absorption score."""

    volume_intensity: VolumeIntensity
    absorption: AbsorptionScore


@dataclass(frozen=True, slots=True)
class Primitives:
    """All ten primitives grouped exactly as documented in the specification."""

    symbol: str
    timeframe: Timeframe
    open_time: dt.datetime
    structure: StructurePrimitives
    direction: DirectionPrimitives
    pressure: PressurePrimitives
    quality: QualityPrimitives

    @property
    def price(self) -> float:
        """Last traded price of the analysed bar."""
        return self.structure.footprint.close

    @property
    def bias(self) -> str:
        """Overall directional bias derived from delta and cumulative delta."""
        score = self.direction.delta.ratio + self.direction.cvd.slope / 10
        if self.direction.delta.direction == BIAS_BULLISH and score > 0:
            return BIAS_BULLISH
        if self.direction.delta.direction == BIAS_BEARISH and score < 0:
            return BIAS_BEARISH
        return BIAS_NEUTRAL


@dataclass(frozen=True, slots=True)
class PrimitiveInput:
    """Everything a primitive computation needs.

    Attributes:
        bar: The bar that just closed.
        history: Previously closed bars, oldest first. Excludes ``bar``.
        book: Order book state at the moment the bar closed.
        config: Primitive thresholds.
        price_decimals: Decimal places used when bucketing price levels.
    """

    bar: FootprintBar
    history: Sequence[FootprintBar] = field(default_factory=tuple)
    book: BookSnapshot = field(default_factory=BookSnapshot.empty)
    config: PrimitivesConfig = field(default_factory=PrimitivesConfig)
    price_decimals: int = 6

    def lookback(self, count: int) -> tuple[FootprintBar, ...]:
        """Return at most ``count`` most recent bars, oldest first."""
        if count <= 0:
            return ()
        return tuple(self.history[-count:])