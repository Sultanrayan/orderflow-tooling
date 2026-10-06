"""Volatility regime and multi timeframe alignment."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import Candle
from ..core.timeframes import Timeframe
from ..primitives.statistics import mean, percentile_rank, safe_div
from .models import (
    ALIGNMENT_ALIGNED,
    ALIGNMENT_CONFLICT,
    ALIGNMENT_MIXED,
    VOLATILITY_EXTREME,
    VOLATILITY_HIGH,
    VOLATILITY_LOW,
    VOLATILITY_NORMAL,
    MultiTimeframe,
    TimeframeState,
    VolatilityState,
)

__all__ = ["build_multi_timeframe", "build_volatility", "timeframe_direction"]

DIRECTION_BULLISH = "bullish"
DIRECTION_BEARISH = "bearish"
DIRECTION_FLAT = "flat"

EXTREME_PERCENTILE = 95.0
HIGH_PERCENTILE = 70.0
LOW_PERCENTILE = 30.0


def build_volatility(candles: Sequence[Candle], lookback: int = 20) -> VolatilityState:
    """Classify the current bar range against its own recent distribution.

    Args:
        candles: Closed candles, oldest first. The last entry is the analysed bar.
        lookback: Bars forming the distribution.

    Returns:
        The volatility regime, the average range used as its scale and the
        percentile of the latest range.
    """
    ranges = [candle.range for candle in candles]
    if not ranges:
        return VolatilityState(regime=VOLATILITY_NORMAL, average_range=0.0, percentile=50.0)

    recent = ranges[-lookback:]
    average_range = mean(recent) or 1e-9
    percentile = percentile_rank(ranges[-1], ranges) * 100
    ratio = safe_div(ranges[-1], average_range, default=1.0)

    if percentile >= EXTREME_PERCENTILE and ratio >= 1.8:
        regime = VOLATILITY_EXTREME
    elif percentile >= HIGH_PERCENTILE:
        regime = VOLATILITY_HIGH
    elif percentile <= LOW_PERCENTILE:
        regime = VOLATILITY_LOW
    else:
        regime = VOLATILITY_NORMAL

    return VolatilityState(
        regime=regime,
        average_range=average_range,
        percentile=percentile,
        atr_ratio=ratio,
    )


def timeframe_direction(candles: Sequence[Candle]) -> str:
    """Direction of a single timeframe based on net move and closes.

    Returns:
        ``bullish``, ``bearish`` or ``flat``.
    """
    if len(candles) < 2:
        return DIRECTION_FLAT

    first_open = candles[0].open
    last_close = candles[-1].close
    change = safe_div(last_close - first_open, first_open or 1.0)
    if change > 0.0005:
        return DIRECTION_BULLISH
    if change < -0.0005:
        return DIRECTION_BEARISH
    return DIRECTION_FLAT


def build_multi_timeframe(
    candles_by_timeframe: dict[Timeframe, Sequence[Candle]],
    minimum_bars: int = 2,
) -> MultiTimeframe:
    """Score agreement across the inspected timeframes.

    Timeframes that have not moved (flat) are treated as neutral: they neither
    support nor oppose the majority. Alignment is ``aligned`` only when every
    directional timeframe points the same way.

    Args:
        candles_by_timeframe: Closed candles per timeframe. Timeframes with
            fewer than ``minimum_bars`` bars are ignored.
        minimum_bars: Bars required for a timeframe to contribute.

    Returns:
        Alignment direction plus a ``0``-``100`` score: the share of inspected
        timeframes pointing the majority way.
    """
    states: list[TimeframeState] = []
    for timeframe in sorted(candles_by_timeframe, key=lambda item: item.seconds):
        candles = candles_by_timeframe[timeframe]
        if len(candles) < minimum_bars:
            continue
        states.append(
            TimeframeState(
                timeframe=timeframe,
                direction=timeframe_direction(candles),
                change_pct=_change_pct(candles),
                bars=len(candles),
            )
        )

    if not states:
        return MultiTimeframe(direction=ALIGNMENT_MIXED, score=0.0)

    bullish = sum(1 for state in states if state.direction == DIRECTION_BULLISH)
    bearish = sum(1 for state in states if state.direction == DIRECTION_BEARISH)
    directional = bullish + bearish
    total = len(states)
    dominant = max(bullish, bearish)

    if directional == 0:
        direction = ALIGNMENT_MIXED
    elif bullish == bearish:
        direction = ALIGNMENT_CONFLICT
    elif directional == dominant:
        direction = ALIGNMENT_ALIGNED
    else:
        direction = ALIGNMENT_MIXED

    score = round(dominant / total * 100)
    return MultiTimeframe(direction=direction, score=score, states=tuple(states))


def _change_pct(candles: Sequence[Candle]) -> float:
    """Net percentage move across the supplied candles."""
    if len(candles) < 2:
        return 0.0
    return safe_div(candles[-1].close - candles[0].open, candles[0].open or 1.0)