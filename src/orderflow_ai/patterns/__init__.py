"""Layer 3: pattern detection and the no-pattern fallback scorer."""

from __future__ import annotations

from .base import PatternDetector
from .engine import PatternEngine
from .fallback import (
    DEFAULT_WEIGHTS,
    FallbackContext,
    FallbackScore,
    FallbackScorer,
    ScoreComponent,
)
from .models import (
    DIRECTION_BEARISH,
    DIRECTION_BULLISH,
    DIRECTION_NEUTRAL,
    TIERS,
    PatternContext,
    PatternMatch,
    PatternResult,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "DIRECTION_BEARISH",
    "DIRECTION_BULLISH",
    "DIRECTION_NEUTRAL",
    "FallbackContext",
    "FallbackScore",
    "FallbackScorer",
    "PatternContext",
    "PatternDetector",
    "PatternEngine",
    "PatternMatch",
    "PatternResult",
    "ScoreComponent",
    "TIERS",
]