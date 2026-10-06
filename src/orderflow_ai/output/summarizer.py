"""Summarisation: turn a full analysis into compact, model friendly facts.

Every method here returns plain JSON-compatible types. Nothing in this module
knows about key shortening or size tiers: it only decides *what* is worth
sending, never *how* it is encoded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..context.models import MarketContext
from ..patterns.models import PatternMatch, PatternResult
from ..primitives.models import FootprintRow, Primitives

if TYPE_CHECKING:  # Imported for typing only: avoids an import cycle.
    from ..core.analysis import Analysis

__all__ = ["Summarizer"]

MAX_FOOTPRINT_LEVELS = 6


class Summarizer:
    """Builds the summarised sections of the payload."""

    def __init__(self, max_patterns: int = 3, max_levels: int = 3) -> None:
        """Create the summariser.

        Args:
            max_patterns: Patterns kept per side of the payload.
            max_levels: Support and resistance levels kept per side.
        """
        self._max_patterns = max_patterns
        self._max_levels = max_levels

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def market(self, analysis: Analysis) -> dict[str, object]:
        """Symbol, timeframe, bar identity and current price."""
        return {
            "symbol": analysis.symbol,
            "timeframe": str(analysis.timeframe),
            "bar_open": analysis.open_time.isoformat(timespec="minutes"),
            "price": analysis.price,
        }

    def direction(self, primitives: Primitives) -> dict[str, object]:
        """Delta, cumulative delta and divergence readings."""
        delta = primitives.direction.delta
        cvd = primitives.direction.cvd
        divergence = primitives.direction.delta_divergence
        return {
            "delta": delta.value,
            "delta_ratio": delta.ratio,
            "direction": delta.direction,
            "delta_strength": delta.strength,
            "cvd": cvd.value,
            "cvd_slope": cvd.slope,
            "momentum": cvd.momentum,
            "divergence": divergence.type if divergence.detected else None,
            "divergence_confidence": divergence.confidence if divergence.detected else None,
        }

    def structure(self, primitives: Primitives, max_footprint_levels: int = MAX_FOOTPRINT_LEVELS) -> dict[str, object]:
        """Footprint, volume profile and value migration."""
        footprint = primitives.structure.footprint
        profile = primitives.structure.volume_profile
        migration = primitives.structure.value_migration
        return {
            "footprint": {
                "open": footprint.open,
                "high": footprint.high,
                "low": footprint.low,
                "close": footprint.close,
                "volume": footprint.total_volume,
                "delta": footprint.delta,
                "trades": footprint.trade_count,
                "levels": [_level_row(row) for row in _top_levels(footprint.levels, max_footprint_levels)],
            },
            "volume_profile": {
                "poc": profile.poc,
                "vah": profile.vah,
                "val": profile.val,
                "hvn": list(profile.hvn[:2]),
                "lvn": list(profile.lvn[:2]),
            },
            "value_migration": {
                "direction": migration.poc_direction,
                "strength": migration.migration_strength,
            },
        }

    def pressure(self, primitives: Primitives) -> dict[str, object]:
        """Imbalance levels and live book metrics."""
        imbalance = primitives.pressure.imbalance
        orderbook = primitives.pressure.orderbook
        return {
            "imbalance": {
                "threshold": imbalance.threshold,
                "count": len(imbalance.levels),
                "strongest": (
                    {
                        "price": imbalance.strongest.price,
                        "side": imbalance.strongest.side.value,
                        "ratio": imbalance.strongest.ratio,
                    }
                    if imbalance.strongest
                    else None
                ),
            },
            "orderbook": {
                "imbalance_ratio": orderbook.imbalance_ratio,
                "direction": orderbook.imbalance_direction,
                "bid_depth": orderbook.bid_depth,
                "ask_depth": orderbook.ask_depth,
                "spread_bps": orderbook.spread_bps,
                "walls": [
                    {
                        "price": wall.price,
                        "side": wall.side.value,
                        "size": wall.size,
                    }
                    for wall in orderbook.walls[:2]
                ],
            },
        }

    def quality(self, primitives: Primitives) -> dict[str, object]:
        """Volume intensity and absorption score."""
        intensity = primitives.quality.volume_intensity
        absorption = primitives.quality.absorption
        return {
            "volume_intensity": {
                "ratio": intensity.ratio,
                "percentile": intensity.percentile,
                "level": intensity.intensity,
            },
            "absorption": {
                "detected": absorption.detected,
                "confidence": absorption.confidence,
                "side": absorption.side.value if absorption.side else None,
                "price": absorption.price,
            },
        }

    def context(self, context: MarketContext) -> dict[str, object]:
        """Structure, session, volatility and timeframe alignment."""
        return {
            "structure": context.structure.label,
            "bos": context.structure.bos,
            "choch": context.structure.choch,
            "session": context.session.primary,
            "active_sessions": list(context.session.active),
            "volatility": {
                "regime": context.volatility.regime,
                "percentile": context.volatility.percentile,
                "average_range": context.volatility.average_range,
            },
            "multi_timeframe": {
                "direction": context.multi_timeframe.direction,
                "score": context.multi_timeframe.score,
            },
            "level_proximity": context.level_proximity,
        }

    def levels(self, context: MarketContext) -> dict[str, object]:
        """Nearest support and resistance plus the point of control."""
        levels = context.levels
        return {
            "support": list(levels.support[: self._max_levels]),
            "resistance": list(levels.resistance[: self._max_levels]),
            "poc": levels.poc,
        }

    def patterns(self, matches: tuple[PatternMatch, ...]) -> list[dict[str, object]]:
        """One entry per pattern, strongest first."""
        return [
            {
                "name": match.name,
                "label": match.label,
                "price": match.price,
                "direction": match.direction,
                "confidence": match.confidence,
                "bars_ago": match.bars_ago,
                "evidence": list(match.evidence),
            }
            for match in matches[: self._max_patterns]
        ]

    def fallback(self, result: PatternResult) -> dict[str, object] | None:
        """Fallback breakdown, only present when no pattern was detected."""
        score = result.fallback
        if score is None:
            return None
        return {
            "score": score.score,
            "tier": score.tier,
            "components": [
                {
                    "name": component.name,
                    "points": round(component.points, 1),
                    "max": component.max_points,
                }
                for component in score.components
            ],
        }


def _level_row(row: FootprintRow) -> dict[str, float]:
    """Encode one footprint level."""
    return {"price": row.price, "bid": row.bid_volume, "ask": row.ask_volume}


def _top_levels(levels: tuple[FootprintRow, ...], limit: int) -> list[FootprintRow]:
    """Keep the busiest levels, then restore price ordering."""
    if limit <= 0 or len(levels) <= limit:
        return list(levels)
    busiest = sorted(levels, key=lambda row: row.total_volume, reverse=True)[:limit]
    return sorted(busiest, key=lambda row: row.price)