"""Reversal family detectors: trapped traders and liquidity sweeps.

Both patterns are failed breakouts; the difference is *who* was aggressive.
Trapped traders requires decisive delta on the breakout bar: aggressive buyers
(or sellers) pushed through a level and were immediately punished. A liquidity
sweep is the opposite: the break happens on weak, absorptive flow, which is what
a stop hunt looks like.
"""

from __future__ import annotations

from ..core.models import FootprintBar
from ..primitives.statistics import clamp
from .base import PatternDetector
from .models import DIRECTION_BEARISH, DIRECTION_BULLISH, PatternContext, PatternMatch

__all__ = ["LiquiditySweepDetector", "TrappedTradersDetector"]

MAX_BARS_AGO = 3
MAX_OVERSHOOT_TICKS = 8.0
TRAPPED_MIN_DELTA_RATIO = 0.2
SWEEP_MAX_DELTA_RATIO = 0.2
# A sweep is a poke, not a breakout: beyond this share of the bar range the
# excursion stops counting as a stop hunt.
SWEEP_MAX_EXTENSION_SHARE = 0.6


class TrappedTradersDetector(PatternDetector):
    """A decisive breakout through a key level that failed immediately."""

    name = "trapped_traders"
    label = "Trapped Traders"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Detect a failed, delta-confirmed breakout of a key level."""
        bar = context.bar
        if bar is None:
            return None

        best: PatternMatch | None = None
        for bars_ago in range(1, min(MAX_BARS_AGO, len(context.bars) - 1) + 1):
            breakout = context.bars[-bars_ago - 1]
            delta_ratio = breakout.delta / (breakout.total_volume or 1.0)

            for level, direction, extension in _failed_breakouts(context, breakout, bars_ago):
                if direction == DIRECTION_BEARISH and delta_ratio <= TRAPPED_MIN_DELTA_RATIO:
                    continue
                if direction == DIRECTION_BULLISH and delta_ratio >= -TRAPPED_MIN_DELTA_RATIO:
                    continue

                reversal = abs(bar.close - breakout.close) / max(context.average_range(), 1e-9)
                confidence = round(
                    clamp(extension / MAX_OVERSHOOT_TICKS) * 40
                    + clamp(abs(delta_ratio) / 0.5) * 35
                    + clamp(reversal / 1.5) * 25
                )
                match = PatternMatch(
                    name=self.name,
                    label=self.label,
                    confidence=float(confidence),
                    direction=direction,
                    price=level,
                    bars_ago=bars_ago,
                    evidence=(
                        f"breakout extended {extension:.1f} ticks through {level:.4f}",
                        f"delta ratio {delta_ratio:.2f} on the breakout bar",
                        f"closed back inside on bar {bars_ago} ago",
                    ),
                    metrics={
                        "extension_ticks": extension,
                        "breakout_delta_ratio": delta_ratio,
                        "reversal_range_ratio": reversal,
                    },
                )
                if best is None or match.confidence > best.confidence:
                    best = match
        return best


class LiquiditySweepDetector(PatternDetector):
    """A brief, low-delta overshoot beyond a key level."""

    name = "liquidity_sweep"
    label = "Liquidity Sweep"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Detect a stop hunt above or below a key level."""
        bar = context.bar
        if bar is None:
            return None

        delta_ratio = bar.delta / (bar.total_volume or 1.0)
        if abs(delta_ratio) > SWEEP_MAX_DELTA_RATIO:
            return None

        best: PatternMatch | None = None
        for level, direction, overshoot in _sweeps(context, bar):
            depth = _reversal_depth(bar, direction)
            extension = _extension_share(bar, level)
            confidence = round(
                clamp(depth / 1.5) * 45
                + clamp(1.0 - abs(delta_ratio) / SWEEP_MAX_DELTA_RATIO) * 30
                + clamp(1.0 - extension / SWEEP_MAX_EXTENSION_SHARE) * 25
            )
            match = PatternMatch(
                name=self.name,
                label=self.label,
                confidence=float(confidence),
                direction=direction,
                price=level,
                bars_ago=0,
                evidence=(
                    f"wicks {overshoot:.1f} ticks beyond {level:.4f} then closes back",
                    f"delta ratio only {delta_ratio:.2f}",
                    f"rejection covers {depth:.2f} bar ranges",
                ),
                metrics={
                    "overshoot_ticks": overshoot,
                    "delta_ratio": delta_ratio,
                    "reversal_range_ratio": depth,
                    "extension_share": extension,
                },
            )
            if best is None or match.confidence > best.confidence:
                best = match
        return best


def _failed_breakouts(
    context: PatternContext, breakout: FootprintBar, bars_ago: int
) -> list[tuple[float, str, float]]:
    """Key levels pierced by ``breakout`` but not held by the latest bar."""
    bar = context.bar
    tick = max(context.tick_size, 1e-9)
    found: list[tuple[float, str, float]] = []

    for level in context.resistances:
        if breakout.high > level and bar is not None and bar.close < level:
            found.append((level, DIRECTION_BEARISH, (breakout.high - level) / tick))
    for level in context.supports:
        if breakout.low < level and bar is not None and bar.close > level:
            found.append((level, DIRECTION_BULLISH, (level - breakout.low) / tick))
    return found


def _sweeps(context: PatternContext, bar: FootprintBar) -> list[tuple[float, str, float]]:
    """Key levels wicked through by ``bar`` but closed back inside."""
    sweeps: list[tuple[float, str, float]] = []
    tick = max(context.tick_size, 1e-9)

    for level in context.resistances:
        overshoot = (bar.high - level) / tick
        if 0 < overshoot <= MAX_OVERSHOOT_TICKS and bar.close < level:
            sweeps.append((level, DIRECTION_BEARISH, overshoot))
    for level in context.supports:
        overshoot = (level - bar.low) / tick
        if 0 < overshoot <= MAX_OVERSHOOT_TICKS and bar.close > level:
            sweeps.append((level, DIRECTION_BULLISH, overshoot))
    return sweeps


def _reversal_depth(bar: FootprintBar, direction: str) -> float:
    """How far price retreated from the swept level, in bar ranges."""
    bar_range = bar.range or 1e-9
    if direction == DIRECTION_BEARISH:
        return (bar.high - bar.close) / bar_range
    return (bar.close - bar.low) / bar_range


def _extension_share(bar: FootprintBar, level: float) -> float:
    """Share of the bar range that sits beyond ``level``."""
    bar_range = bar.range or 1e-9
    if level > bar.low:
        return max(0.0, bar.high - level) / bar_range
    return max(0.0, level - bar.low) / bar_range