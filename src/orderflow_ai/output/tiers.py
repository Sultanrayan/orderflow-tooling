"""Tiered prompt selection.

The specification defines three payload sizes. The rule is deliberately simple:
strong, corroborated evidence earns a tiny payload, weak or absent evidence
earns a large one, because that is exactly when the model needs the raw detail.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.analysis import Analysis

__all__ = [
    "TIER_FULL",
    "TIER_MINIMAL",
    "TIER_STANDARD",
    "TIER_TARGET_TOKENS",
    "TierSelector",
    "TierSummary",
]

TIER_MINIMAL = "minimal"
TIER_STANDARD = "standard"
TIER_FULL = "full"

TIER_TARGET_TOKENS = {TIER_MINIMAL: 150, TIER_STANDARD: 500, TIER_FULL: 1500}

MINIMAL_MIN_PATTERNS = 2
MINIMAL_MIN_CONFIDENCE = 80.0
FULL_MAX_CONFIDENCE = 50.0


@dataclass(frozen=True, slots=True)
class TierSummary:
    """Chosen tier and the reason behind it."""

    tier: str
    reason: str

    @property
    def target_tokens(self) -> int:
        """Token budget this tier aims for."""
        return TIER_TARGET_TOKENS[self.tier]


class TierSelector:
    """Chooses the payload tier from the analysis and configuration."""

    def __init__(self, configured: str = "auto") -> None:
        """Create the selector.

        Args:
            configured: ``auto`` to derive the tier, or a fixed tier name.
        """
        self._configured = (configured or "auto").lower()

    @property
    def configured(self) -> str:
        """Configured mode: ``auto`` or a fixed tier."""
        return self._configured

    def select(self, analysis: Analysis) -> TierSummary:
        """Pick the tier for ``analysis``."""
        if self._configured != "auto":
            return TierSummary(tier=self._configured, reason=f"fixed tier {self._configured}")

        patterns = analysis.patterns
        confidence = patterns.confidence

        if not patterns.detected:
            return TierSummary(
                tier=TIER_FULL,
                reason=f"no pattern detected, fallback score {confidence:.0f}",
            )
        if confidence < FULL_MAX_CONFIDENCE:
            return TierSummary(
                tier=TIER_FULL,
                reason=f"pattern confidence {confidence:.0f} below {FULL_MAX_CONFIDENCE:.0f}",
            )
        if len(patterns.patterns) >= MINIMAL_MIN_PATTERNS and confidence >= MINIMAL_MIN_CONFIDENCE:
            return TierSummary(
                tier=TIER_MINIMAL,
                reason=f"{len(patterns.patterns)} patterns, confidence {confidence:.0f}",
            )
        return TierSummary(
            tier=TIER_STANDARD,
            reason=f"single pattern, confidence {confidence:.0f}",
        )