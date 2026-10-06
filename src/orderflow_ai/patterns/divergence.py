"""Delta divergence detector.

Thin wrapper over the delta divergence primitive: the primitive already measures
price and delta disagreeing, so the detector only translates it into a pattern
with the direction the divergence implies.
"""

from __future__ import annotations

from .base import PatternDetector
from .models import DIRECTION_BEARISH, DIRECTION_BULLISH, PatternContext, PatternMatch

__all__ = ["DeltaDivergenceDetector"]

DIVERGENCE_BEARISH = "bearish"
DIVERGENCE_BULLISH = "bullish"


class DeltaDivergenceDetector(PatternDetector):
    """Price and delta moving in opposite directions over the lookback."""

    name = "delta_divergence"
    label = "Delta Divergence"

    def detect(self, context: PatternContext) -> PatternMatch | None:
        """Translate the delta divergence primitive into a pattern match."""
        divergence = context.primitives.direction.delta_divergence
        if not divergence.detected or divergence.type is None:
            return None

        # A bearish divergence (price up, delta down) points down.
        direction = (
            DIRECTION_BEARISH if divergence.type == DIVERGENCE_BEARISH else DIRECTION_BULLISH
        )
        return PatternMatch(
            name=self.name,
            label=self.label,
            confidence=float(divergence.confidence),
            direction=direction,
            price=context.primitives.price,
            evidence=(
                f"price {divergence.price_change * 100:+.2f}% over {divergence.bars_observed} bars",
                f"cumulative delta {divergence.delta_change:+.1f} against the move",
                f"{divergence.type} divergence",
            ),
            metrics={
                "price_change": divergence.price_change,
                "delta_change": divergence.delta_change,
                "bars_observed": float(divergence.bars_observed),
            },
        )