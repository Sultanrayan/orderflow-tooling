"""Assembly of the layer 4 context.

The builder is the only place that knows how swings, volume profile, session
clock and higher timeframe candles fit together. It depends on layer 2 for the
primitives and never on layer 3.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from ..core.models import Candle, FootprintBar
from ..core.timeframes import Timeframe
from ..primitives.models import Primitives
from .levels import build_key_levels
from .models import MarketContext
from .sessions import build_session_context
from .structure import build_structure, detect_swings
from .volatility import build_multi_timeframe, build_volatility

__all__ = ["ContextBuilder"]

FRACTAL_STRENGTH = 2
LEVEL_TOLERANCE_TICKS = 2.0


class ContextBuilder:
    """Builds a :class:`MarketContext` for a freshly closed bar."""

    def __init__(
        self,
        max_levels: int = 3,
        fractal_strength: int = FRACTAL_STRENGTH,
        volatility_lookback: int = 20,
    ) -> None:
        """Create the builder.

        Args:
            max_levels: Support and resistance levels kept per side.
            fractal_strength: Bars required on each side of a swing pivot.
            volatility_lookback: Bars forming the volatility distribution.
        """
        self._max_levels = max_levels
        self._fractal_strength = fractal_strength
        self._volatility_lookback = volatility_lookback

    def build(
        self,
        primitives: Primitives,
        bars: Sequence[FootprintBar],
        context_candles: dict[Timeframe, Sequence[Candle]] | None = None,
        now: dt.datetime | None = None,
    ) -> MarketContext:
        """Assemble the full context.

        Args:
            primitives: Primitives of the bar that just closed.
            bars: Closed bars, oldest first, including the analysed bar.
            context_candles: Closed candles per higher timeframe.
            now: Moment used for the session clock. Defaults to the bar open time.

        Returns:
            The market context.
        """
        candles = [bar.to_candle() for bar in bars]
        price = primitives.price
        profile = primitives.structure.volume_profile
        open_time = primitives.open_time

        swings = detect_swings(candles, self._fractal_strength)
        structure = build_structure(candles, self._fractal_strength)
        levels = build_key_levels(
            price=price,
            profile=profile,
            swings=swings,
            max_levels=self._max_levels,
            tolerance=LEVEL_TOLERANCE_TICKS * _tick_size(bars),
        )
        session = build_session_context(now or open_time)
        volatility = build_volatility(candles, self._volatility_lookback)
        multi_timeframe = build_multi_timeframe(
            {timeframe: list(values) for timeframe, values in (context_candles or {}).items()}
        )

        return MarketContext(
            symbol=primitives.symbol,
            timeframe=primitives.timeframe,
            open_time=open_time,
            price=price,
            structure=structure,
            levels=levels,
            session=session,
            multi_timeframe=multi_timeframe,
            volatility=volatility,
        )


def _tick_size(bars: Sequence[FootprintBar]) -> float:
    """Smallest price increment observed across the supplied bars."""
    steps = set()
    for bar in bars:
        prices = sorted(bar.levels)
        steps.update(
            round(abs(second - first), 10)
            for first, second in zip(prices, prices[1:], strict=False)
            if abs(second - first) > 0
        )
    return min(steps) if steps else 0.01