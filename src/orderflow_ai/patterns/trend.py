"""Exhaustion detector.

Trend momentum fades when volume and delta stop confirming the move. The
detector requires a trend that is old enough to matter, then looks for a
decline in both size and aggression while price still extends, which is the
classic signature of late trend participants.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import FootprintBar
from ..primitives.statistics import clamp
from .base import PatternDetector
from .models import DIRECTION_BEARISH, DIRECTION_BULLISH, PatternContext, PatternMatch

__all__ = ["ExhaustionDetector"]

MIN_TREND_BARS = 4
VOLUME_DECAY = 0.85
DELTA_DECAY = 0.6


class ExhaustionDetector(PatternDetector):
    """Declining volume and delta at the end of an extended move."""

    name = "exhaustion"
    label = "Exhaustion"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Detect fading momentum behind an extended directional move."""
        bars = context.lookback(MIN_TREND_BARS + 1)
        if len(bars) < MIN_TREND_BARS + 1:
            return None

        direction = _trend_direction(bars)
        if direction is None:
            return None

        trend_bars = bars[:-1]
        last = bars[-1]
        volumes = [bar.total_volume for bar in trend_bars]
        deltas = [bar.delta for bar in trend_bars]

        average_volume = sum(volumes) / len(volumes) or 1.0
        average_delta = sum(deltas) / len(deltas)
        if average_delta == 0:
            return None

        volume_decay = 1.0 - last.total_volume / average_volume
        delta_decay = 1.0 - last.delta / average_delta
        if volume_decay < 1.0 - VOLUME_DECAY or delta_decay < 1.0 - DELTA_DECAY:
            return None

        trend_bars_used = len(bars) - 1
        confidence = round(
            clamp(trend_bars_used / 5.0) * 40
            + clamp(volume_decay / 0.6) * 30
            + clamp(delta_decay) * 30
        )

        return PatternMatch(
            name=self.name,
            label=self.label,
            confidence=float(confidence),
            # Momentum faded behind an up trend, so the expectation is a turn down.
            direction=DIRECTION_BEARISH if direction == DIRECTION_BULLISH else DIRECTION_BULLISH,
            price=last.close,
            evidence=(
                f"{trend_bars_used} bar {direction} trend still extending",
                f"volume {volume_decay * 100:.0f}% below trend average",
                f"delta {delta_decay * 100:.0f}% weaker than trend average",
            ),
            metrics={
                "trend_bars": float(trend_bars_used),
                "volume_decay": volume_decay,
                "delta_decay": delta_decay,
            },
        )


def _trend_direction(bars: Sequence[FootprintBar]) -> str | None:
    """Classify the bars as an up, down or flat sequence."""
    closes = [bar.close for bar in bars]
    steps = len(closes) - 1
    rising = sum(1 for first, second in zip(closes, closes[1:], strict=False) if second > first)
    falling = sum(1 for first, second in zip(closes, closes[1:], strict=False) if second < first)
    needed = max(1, steps * 0.75)

    if rising >= needed:
        return DIRECTION_BULLISH
    if falling >= needed:
        return DIRECTION_BEARISH
    return None