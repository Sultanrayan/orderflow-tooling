"""Evidence prioritisation.

The model never needs everything. :class:`Prioritizer` ranks the evidence so the
payload leads with what actually drives the decision, and reports what was left
out so a human can tell the difference between "no signal" and "not shipped".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analysis import Analysis
from .summarizer import Summarizer

__all__ = ["Prioritisation", "Prioritizer"]

TIER_WEIGHTS = {"minimal": 1.0, "standard": 0.7, "full": 0.4}


@dataclass(frozen=True, slots=True)
class Prioritisation:
    """Outcome of a prioritisation pass."""

    lead: str
    summary: str
    included: tuple[str, ...]
    dropped: tuple[str, ...]

    @property
    def dropped_count(self) -> int:
        """How many evidence items were left out of the payload."""
        return len(self.dropped)


class Prioritizer:
    """Scores evidence items and decides which ones lead the payload."""

    def __init__(self, summarizer: Summarizer) -> None:
        """Create the prioritiser.

        Args:
            summarizer: Summariser used to build the compact evidence strings.
        """
        self._summarizer = summarizer

    def prioritise(self, analysis: Analysis, tier: str = "standard") -> Prioritisation:
        """Rank the evidence for ``analysis``.

        Args:
            analysis: The analysis to rank.
            tier: Payload tier, used to weight pattern evidence.

        Returns:
            The leading evidence, a one line summary and the included and
            dropped item names.
        """
        weight = TIER_WEIGHTS.get(tier, 0.7)
        scores = self._scores(analysis, weight)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        included = tuple(name for name, _ in ranked if _ > 0)
        dropped = tuple(name for name, _ in ranked if _ <= 0)
        lead = included[0] if included else "none"
        return Prioritisation(
            lead=lead,
            summary=self._summary(analysis, lead),
            included=included,
            dropped=dropped,
        )

    def _scores(self, analysis: Analysis, weight: float) -> dict[str, float]:
        """Score each evidence family on a 0-1 scale."""
        primitives = analysis.primitives
        patterns = analysis.patterns

        pattern_score = weight * (patterns.confidence / 100 if patterns.detected else 0.0)
        delta_score = min(abs(primitives.direction.delta.ratio) / 0.4, 1.0)
        cvd_score = min(abs(primitives.direction.cvd.slope), 1.0)
        absorption_score = primitives.quality.absorption.confidence / 100
        book_score = min(abs(primitives.pressure.orderbook.imbalance_ratio - 1.0), 1.0)
        mtf_score = analysis.context.multi_timeframe.score / 100
        level_score = analysis.context.level_proximity

        return {
            "patterns": pattern_score,
            "delta": delta_score,
            "cvd": cvd_score,
            "absorption": absorption_score,
            "orderbook": book_score,
            "multi_timeframe": mtf_score,
            "levels": level_score,
        }

    def _summary(self, analysis: Analysis, lead: str) -> str:
        """One line describing the leading evidence."""
        best = analysis.patterns.best
        if best is not None:
            return (
                f"{best.label} at {best.price:.6g} ({best.direction}, "
                f"confidence {best.confidence:.0f})"
            )
        if analysis.patterns.fallback is not None:
            fallback = analysis.patterns.fallback
            return f"no pattern, primitives grade {fallback.score:.0f} ({fallback.tier})"
        return f"no evidence scored ({lead})"