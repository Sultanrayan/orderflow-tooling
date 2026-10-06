"""Pattern engine: runs every detector, filters and ranks the results.

Ranking is deterministic and explainable: confidence first, then recency, then
name. When nothing survives the filter the fallback scorer supplies a
confidence so downstream layers always receive a graded opinion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

from ..core.config import PatternsConfig
from .absorption import AbsorptionDetector, IcebergDetector
from .auction import UnfinishedAuctionDetector
from .base import PatternDetector
from .divergence import DeltaDivergenceDetector
from .fallback import FallbackContext, FallbackScorer
from .models import PatternContext, PatternMatch, PatternResult, dedupe
from .reversal import LiquiditySweepDetector, TrappedTradersDetector
from .trend import ExhaustionDetector

__all__ = ["PatternEngine", "default_detectors"]

logger = logging.getLogger(__name__)

MAX_DETECTORS: Final[int] = 7


def default_detectors() -> tuple[PatternDetector, ...]:
    """Instantiate the seven documented detectors."""
    return (
        AbsorptionDetector(),
        TrappedTradersDetector(),
        LiquiditySweepDetector(),
        DeltaDivergenceDetector(),
        ExhaustionDetector(),
        IcebergDetector(),
        UnfinishedAuctionDetector(),
    )


class PatternEngine:
    """Detects order flow patterns and grades the current read."""

    def __init__(
        self,
        config: PatternsConfig | None = None,
        detectors: Sequence[PatternDetector] | None = None,
        scorer: FallbackScorer | None = None,
    ) -> None:
        """Create the engine.

        Args:
            config: Confidence and count filters.
            detectors: Detectors to run. Defaults to :func:`default_detectors`.
            scorer: Fallback scorer. A default one is created when omitted.
        """
        if detectors is not None and len(detectors) > MAX_DETECTORS:
            raise ValueError(f"at most {MAX_DETECTORS} detectors are supported")
        self._config = config or PatternsConfig()
        self._detectors = tuple(detectors) if detectors is not None else default_detectors()
        self._scorer = scorer or FallbackScorer()

    @property
    def config(self) -> PatternsConfig:
        """Filters in use."""
        return self._config

    @property
    def detector_names(self) -> tuple[str, ...]:
        """Names of the registered detectors."""
        return tuple(detector.name for detector in self._detectors)

    def detect(
        self,
        context: PatternContext,
        fallback_context: FallbackContext | None = None,
    ) -> PatternResult:
        """Run every detector and return the filtered, ranked result.

        Args:
            context: Inputs for the detectors.
            fallback_context: Layer 4 readings used when no pattern survives.

        Returns:
            The accepted patterns and, when the list is empty, the fallback
            score that explains the absence of a pattern.
        """
        matches = self._run_detectors(context)
        accepted = self._accept(matches)

        if accepted:
            return PatternResult(patterns=accepted, considered=len(matches))

        fallback = self._scorer.score(context.primitives, fallback_context)
        logger.debug(
            "no pattern detected (score %.0f, tier %s)", fallback.score, fallback.tier
        )
        return PatternResult(patterns=(), fallback=fallback, considered=len(matches))

    def _run_detectors(self, context: PatternContext) -> list[PatternMatch]:
        """Collect matches, treating detector errors as "no pattern"."""
        matches: list[PatternMatch] = []
        for detector in self._detectors:
            try:
                match = detector.detect(context)
            except Exception as exc:  # noqa: BLE001 - one bad detector must not stop analysis
                logger.warning("detector %s failed: %s", detector.name, exc)
                continue
            if match is not None:
                matches.append(match)
        return matches

    def _accept(self, matches: list[PatternMatch]) -> tuple[PatternMatch, ...]:
        """Drop stale or low confidence matches, then rank and cap them."""
        fresh = [
            match
            for match in matches
            if match.confidence >= self._config.min_confidence
            and match.bars_ago <= self._config.max_age_bars
        ]
        ranked = sorted(dedupe(fresh), key=_rank_key)
        return tuple(ranked[: self._config.max_patterns])


def _rank_key(match: PatternMatch) -> tuple[float, int, str]:
    """Sort key: strongest first, then most recent, then alphabetical."""
    return (-match.confidence, match.bars_ago, match.name)