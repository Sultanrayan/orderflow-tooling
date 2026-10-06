"""Layer 4: market structure, key levels, sessions and volatility."""

from __future__ import annotations

from .builder import ContextBuilder
from .levels import build_key_levels, round_numbers
from .models import (
    KeyLevels,
    MarketContext,
    MarketStructure,
    MultiTimeframe,
    SessionContext,
    TimeframeState,
    VolatilityState,
)
from .sessions import Session, build_session_context, session_of
from .structure import Swing, build_structure, detect_swings
from .volatility import build_multi_timeframe, build_volatility

__all__ = [
    "ContextBuilder",
    "KeyLevels",
    "MarketContext",
    "MarketStructure",
    "MultiTimeframe",
    "Session",
    "SessionContext",
    "Swing",
    "TimeframeState",
    "VolatilityState",
    "build_key_levels",
    "build_multi_timeframe",
    "build_session_context",
    "build_structure",
    "build_volatility",
    "detect_swings",
    "round_numbers",
    "session_of",
]