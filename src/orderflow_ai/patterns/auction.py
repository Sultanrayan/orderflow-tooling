"""Unfinished auction detector.

When a bar closes on an extreme where one side never traded at all, the auction
at that price did not finish: sellers were never given a chance to respond (or
buyers were). Such a bar usually returns to balance and, more importantly, back
through the price that was left behind.
"""

from __future__ import annotations

from ..core.models import FootprintBar, FootprintLevel
from ..primitives.statistics import clamp
from .base import PatternDetector
from .models import DIRECTION_BEARISH, DIRECTION_BULLISH, PatternContext, PatternMatch

__all__ = ["UnfinishedAuctionDetector"]

MIN_BAR_SHARE = 0.35
MIN_VOLUME = 1.0


class UnfinishedAuctionDetector(PatternDetector):
    """One sided auction at the extreme of a bar."""

    name = "unfinished_auction"
    label = "Unfinished Auction"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Detect a bar whose extreme price was never offered, or never bid."""
        bar = context.bar
        if bar is None or not bar.levels or bar.total_volume < MIN_VOLUME:
            return None

        high_level = bar.level_at(bar.high, context.price_decimals)
        low_level = bar.level_at(bar.low, context.price_decimals)
        high_share = _share(high_level, bar)
        low_share = _share(low_level, bar)

        # No offers were lifted at the high: buyers were paid to leave.
        if high_level is not None and high_level.ask_volume == 0 and high_share >= MIN_BAR_SHARE:
            confidence = round(55 + clamp(high_share) * 40)
            return PatternMatch(
                name=self.name,
                label=self.label,
                confidence=float(confidence),
                direction=DIRECTION_BEARISH,
                price=bar.high,
                evidence=(
                    "no offers traded at the bar high",
                    f"{high_share * 100:.0f}% of volume below the high",
                    "buy side of the auction unfinished",
                ),
                metrics={"high_share": high_share},
            )

        # No bids were hit at the low: sellers were never served.
        if low_level is not None and low_level.bid_volume == 0 and low_share >= MIN_BAR_SHARE:
            confidence = round(55 + clamp(low_share) * 40)
            return PatternMatch(
                name=self.name,
                label=self.label,
                confidence=float(confidence),
                direction=DIRECTION_BULLISH,
                price=bar.low,
                evidence=(
                    "no bids traded at the bar low",
                    f"{low_share * 100:.0f}% of volume above the low",
                    "sell side of the auction unfinished",
                ),
                metrics={"low_share": low_share},
            )
        return None


def _share(level: FootprintLevel | None, bar: FootprintBar) -> float:
    """Volume traded at ``level`` as a share of the bar."""
    if level is None or bar.total_volume <= 0:
        return 0.0
    return level.total_volume / bar.total_volume