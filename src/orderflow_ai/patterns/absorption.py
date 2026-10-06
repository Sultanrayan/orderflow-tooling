"""Absorption family detectors: classic absorption and iceberg reveals.

Both patterns describe volume that did not move price. Absorption is about size
against a compressed range; an iceberg is about *how* that size arrived, in many
small clips at a single level while the price refuses to move through it.
"""

from __future__ import annotations

from ..core.models import FootprintBar, FootprintLevel
from ..primitives.statistics import clamp
from .base import PatternDetector
from .models import DIRECTION_BEARISH, DIRECTION_BULLISH, PatternContext, PatternMatch

__all__ = ["AbsorptionDetector", "IcebergDetector"]

ICEBERG_MIN_SHARE = 0.25
ICEBERG_MIN_TRADES = 4
ICEBERG_MAX_CLIP_RATIO = 0.6
ICEBERG_EDGE_SHARE = 0.25


class AbsorptionDetector(PatternDetector):
    """Heavy volume, decisive delta and a compressed range in one bar."""

    name = "absorption"
    label = "Absorption"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Detect absorbed aggressive volume at a single price level."""
        absorption = context.primitives.quality.absorption
        if not absorption.detected or absorption.price is None:
            return None

        side = absorption.side
        if side is None:
            return None

        level = context.level_at(absorption.price)
        level_share = 0.0
        if level is not None and context.bar is not None and context.bar.total_volume > 0:
            level_share = level.total_volume / context.bar.total_volume

        return PatternMatch(
            name=self.name,
            label=self.label,
            confidence=float(absorption.confidence),
            direction=DIRECTION_BULLISH if side.value == "buy" else DIRECTION_BEARISH,
            price=absorption.price,
            evidence=(
                f"volume {absorption.volume_ratio:.2f}x average",
                f"range {absorption.range_ratio:.2f}x average",
                f"{level_share * 100:.0f}% of bar volume at one level",
            ),
            metrics={
                "volume_ratio": absorption.volume_ratio,
                "range_ratio": absorption.range_ratio,
                "level_share": level_share,
            },
        )


class IcebergDetector(PatternDetector):
    """Repeated small clips at one level betraying a hidden large order."""

    name = "iceberg"
    label = "Iceberg"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Detect a level refilled by many small trades with no price progress."""
        bar = context.bar
        if bar is None or bar.total_volume <= 0 or bar.trade_count < ICEBERG_MIN_TRADES:
            return None

        bar_range = bar.range or 1e-9
        best_level = None
        best_score = -1.0

        for level in bar.sorted_levels:
            share = level.total_volume / bar.total_volume
            if share < ICEBERG_MIN_SHARE or level.trade_count < ICEBERG_MIN_TRADES:
                continue
            # A hidden order only makes sense at an extreme of the bar.
            edge_distance = min(abs(level.price - bar.high), abs(level.price - bar.low))
            if edge_distance / bar_range > ICEBERG_EDGE_SHARE:
                continue
            clip_ratio = _clip_ratio(level, bar)
            if clip_ratio > ICEBERG_MAX_CLIP_RATIO:
                continue
            score = share * (1.0 - clip_ratio)
            if score > best_score:
                best_level, best_score = level, score

        if best_level is None:
            return None

        share = best_level.total_volume / bar.total_volume
        clip_ratio = _clip_ratio(best_level, bar)
        confidence = round(clamp(share / 0.5) * 60 + clamp(1.0 - clip_ratio) * 40)
        at_high = abs(best_level.price - bar.high) <= abs(best_level.price - bar.low)

        return PatternMatch(
            name=self.name,
            label=self.label,
            confidence=float(confidence),
            direction=DIRECTION_BEARISH if at_high else DIRECTION_BULLISH,
            price=best_level.price,
            evidence=(
                f"{share * 100:.0f}% of bar volume on one level",
                f"{best_level.trade_count} small clips, price unmoved",
                "hidden offer" if at_high else "hidden bid",
            ),
            metrics={"level_share": share, "clip_ratio": clip_ratio},
        )


def _clip_ratio(level: FootprintLevel, bar: FootprintBar) -> float:
    """Average clip at ``level`` relative to the largest clip elsewhere in ``bar``.

    Without another level in the bar there is nothing to compare against, so the
    ratio is reported as ``0.0`` and the size comparison is skipped.
    """
    other_clip = max(
        (
            candidate.average_trade_size
            for candidate in bar.levels.values()
            if candidate.price != level.price
        ),
        default=0.0,
    )
    return level.average_trade_size / other_clip if other_clip > 0 else 0.0