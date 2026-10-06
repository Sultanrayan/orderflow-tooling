"""The analysis object handed between layers and to layer 5.

:class:`Analysis` is the single immutable snapshot of one closed bar. Keeping it
as one object means the output layer, the AI client and the backtester all see
exactly the same evidence, which is what makes a signal reproducible.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..context.models import MarketContext
from ..core.timeframes import Timeframe
from ..patterns.models import PatternResult
from ..primitives.models import Primitives

__all__ = ["Analysis"]


@dataclass(frozen=True, slots=True)
class Analysis:
    """One complete analysis of a closed bar.

    Attributes:
        symbol: Traded symbol.
        timeframe: Candle timeframe the analysis is based on.
        open_time: Bar open time in UTC.
        price: Last traded price of the bar.
        primitives: Layer 2 output.
        patterns: Layer 3 output.
        context: Layer 4 output.
        bars: Closed bars used, oldest first. Kept for debugging and replay.
    """

    symbol: str
    timeframe: Timeframe
    open_time: dt.datetime
    price: float
    primitives: Primitives
    patterns: PatternResult
    context: MarketContext
    bars: tuple = ()

    @property
    def confidence(self) -> float:
        """Confidence carried into the prompt: pattern or fallback confidence."""
        return self.patterns.confidence

    @property
    def direction(self) -> str:
        """Direction implied by the strongest evidence."""
        if self.patterns.patterns:
            return self.patterns.patterns[0].direction
        return self.primitives.bias

    @property
    def has_pattern(self) -> bool:
        """``True`` when at least one pattern was detected."""
        return self.patterns.detected

    @property
    def pattern_names(self) -> tuple[str, ...]:
        """Names of the detected patterns."""
        return self.patterns.names