"""Value objects for pattern detection.

Layer 3 deliberately does not import layer 4. Everything a detector needs about
market context (nearest support and resistance) arrives as plain price tuples in
:class:`PatternContext`, which the pipeline fills from the layer 4 result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.models import FootprintBar
from ..primitives.models import Primitives

if TYPE_CHECKING:  # Imported for typing only: avoids an import cycle.
    from ..core.models import FootprintLevel
    from .fallback import FallbackScore

__all__ = [
    "DIRECTION_BEARISH",
    "DIRECTION_BULLISH",
    "DIRECTION_NEUTRAL",
    "PatternContext",
    "PatternMatch",
    "PatternResult",
    "TIERS",
    "classify_tier",
]

DIRECTION_BULLISH = "bullish"
DIRECTION_BEARISH = "bearish"
DIRECTION_NEUTRAL = "neutral"

TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_NONE = "none"
TIERS = (TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_NONE)


@dataclass(frozen=True, slots=True)
class PatternMatch:
    """One detected pattern.

    Attributes:
        name: Stable machine name, for example ``absorption``.
        label: Human readable name for logs and reports.
        confidence: Confidence between 0 and 100.
        direction: Expected direction of the setup the pattern implies.
        price: Price the pattern is anchored to.
        bars_ago: How many bars ago the pattern completed. ``0`` is the latest.
        evidence: Short, human readable statements backing the confidence.
        metrics: Numeric inputs behind the score, shipped in the payload.
    """

    name: str
    label: str
    confidence: float
    direction: str
    price: float
    bars_ago: int = 0
    evidence: tuple[str, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> tuple[str, float, str, float, int]:
        """Compact ``(name, price, direction, confidence, bars_ago)`` encoding."""
        return (self.name, self.price, self.direction, self.confidence, self.bars_ago)


@dataclass(frozen=True, slots=True)
class PatternContext:
    """Everything a detector may look at.

    Attributes:
        primitives: Primitives of the bar that just closed.
        bars: Recent closed bars, oldest first. The last entry is the analysed
            bar.
        supports: Known support prices, nearest first.
        resistances: Known resistance prices, nearest first.
        tick_size: Smallest price increment, used to scale overshoots.
        price_decimals: Decimal places used when looking up footprint levels.
    """

    primitives: Primitives
    bars: tuple[FootprintBar, ...]
    supports: tuple[float, ...] = ()
    resistances: tuple[float, ...] = ()
    tick_size: float = 0.01
    price_decimals: int = 6

    @property
    def bar(self) -> FootprintBar | None:
        """The bar that just closed."""
        return self.bars[-1] if self.bars else None

    @property
    def history(self) -> tuple[FootprintBar, ...]:
        """Bars before the analysed bar, oldest first."""
        return self.bars[:-1]

    def lookback(self, count: int) -> tuple[FootprintBar, ...]:
        """The ``count`` most recent bars, oldest first."""
        return self.bars[-count:] if count > 0 else ()

    def closes(self, count: int) -> list[float]:
        """Close prices of the ``count`` most recent bars, oldest first."""
        return [bar.close for bar in self.lookback(count)]

    def average_range(self, count: int = 10) -> float:
        """Mean bar range over the lookback window."""
        ranges = [bar.range for bar in self.lookback(count)]
        return sum(ranges) / len(ranges) if ranges else 0.0

    def level_at(self, price: float) -> FootprintLevel | None:
        """Return the footprint level of the latest bar at ``price``."""
        bar = self.bar
        return bar.level_at(price, self.price_decimals) if bar else None


def classify_tier(confidence: float) -> str:
    """Map a confidence score onto the documented confidence tiers."""
    if confidence >= 75:
        return TIER_HIGH
    if confidence >= 50:
        return TIER_MEDIUM
    if confidence > 0:
        return TIER_LOW
    return TIER_NONE


@dataclass(frozen=True, slots=True)
class PatternResult:
    """Result of one pattern detection pass.

    Attributes:
        patterns: Accepted patterns, strongest first.
        fallback: Fallback score, only present when no pattern was detected.
        considered: Number of patterns the detectors matched before filtering.
    """

    patterns: tuple[PatternMatch, ...] = ()
    fallback: FallbackScore | None = None
    considered: int = 0

    @property
    def detected(self) -> bool:
        """``True`` when at least one pattern passed the confidence filter."""
        return bool(self.patterns)

    @property
    def names(self) -> tuple[str, ...]:
        """Machine names of the accepted patterns."""
        return tuple(pattern.name for pattern in self.patterns)

    @property
    def best(self) -> PatternMatch | None:
        """Strongest accepted pattern."""
        return self.patterns[0] if self.patterns else None

    @property
    def confidence(self) -> float:
        """Confidence used downstream: pattern confidence or fallback score."""
        if self.patterns:
            return self.patterns[0].confidence
        return self.fallback.score if self.fallback else 0.0

    @property
    def direction(self) -> str:
        """Direction implied by the strongest evidence."""
        if self.patterns:
            return self.patterns[0].direction
        return DIRECTION_NEUTRAL

    def filter_by_confidence(self, minimum: float) -> tuple[PatternMatch, ...]:
        """Return only the patterns at or above ``minimum``."""
        return tuple(pattern for pattern in self.patterns if pattern.confidence >= minimum)


def dedupe(patterns: Sequence[PatternMatch]) -> tuple[PatternMatch, ...]:
    """Drop repeated patterns, keeping the strongest of each name and price."""
    strongest: dict[tuple[str, float], PatternMatch] = {}
    for pattern in patterns:
        key = (pattern.name, pattern.price)
        current = strongest.get(key)
        if current is None or pattern.confidence > current.confidence:
            strongest[key] = pattern
    return tuple(strongest.values())