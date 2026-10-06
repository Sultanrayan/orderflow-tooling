"""No-pattern fallback confidence.

When no discrete pattern is detected the engine still has to answer the
question "how strong is the case for a trade right now?". The fallback scorer
sums six independent components, each capped at its documented weight, so the
total is directly comparable with a pattern confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..primitives.models import Primitives
from ..primitives.statistics import clamp
from .models import classify_tier

__all__ = [
    "DEFAULT_WEIGHTS",
    "FallbackContext",
    "FallbackScore",
    "FallbackScorer",
    "ScoreComponent",
]

DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "delta_strength": 20.0,
    "orderbook_imbalance": 20.0,
    "mtf_alignment": 20.0,
    "volume_intensity": 15.0,
    "level_proximity": 15.0,
    "volatility_regime": 10.0,
}

VOLATILITY_POINTS = {"low": 0.3, "normal": 1.0, "high": 0.7, "extreme": 0.4}
VOLATILITY_DEFAULT = 0.5


@dataclass(frozen=True, slots=True)
class FallbackContext:
    """Layer 4 readings used by the fallback scorer.

    Attributes:
        mtf_alignment: Agreement of the higher timeframes, ``0.0`` to ``1.0``.
        level_proximity: Distance to the nearest key level, ``0.0`` (far away)
            to ``1.0`` (sitting on the level).
        volatility_regime: ``low``, ``normal``, ``high`` or ``extreme``.
    """

    mtf_alignment: float = 0.0
    level_proximity: float = 0.0
    volatility_regime: str = "normal"


@dataclass(frozen=True, slots=True)
class ScoreComponent:
    """One line of the fallback breakdown."""

    name: str
    points: float
    max_points: float
    detail: str = ""

    @property
    def ratio(self) -> float:
        """Fraction of the available points this component earned."""
        return clamp(self.points / self.max_points) if self.max_points else 0.0


@dataclass(frozen=True, slots=True)
class FallbackScore:
    """Aggregate fallback confidence.

    Attributes:
        score: Total points earned, ``0`` to ``100``.
        tier: Confidence tier derived from ``score``.
        components: Per component breakdown, in weight order.
    """

    score: float
    tier: str
    components: tuple[ScoreComponent, ...] = field(default_factory=tuple)


class FallbackScorer:
    """Scores confidence from primitives when no pattern fired."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Create the scorer.

        Args:
            weights: Component weights. Defaults to :data:`DEFAULT_WEIGHTS`,
                which sums to 100.
        """
        self._weights = dict(weights or DEFAULT_WEIGHTS)

    @property
    def weights(self) -> dict[str, float]:
        """Component weights in use."""
        return dict(self._weights)

    def score(self, primitives: Primitives, context: FallbackContext | None = None) -> FallbackScore:
        """Compute the fallback confidence.

        Args:
            primitives: Primitives of the bar that just closed.
            context: Layer 4 readings. Missing values score zero, so a caller
                without context still gets a usable primitives-only score.

        Returns:
            The aggregate score with its component breakdown.
        """
        context = context or FallbackContext()
        components = (
            self._delta_strength(primitives),
            self._orderbook_imbalance(primitives),
            self._mtf_alignment(context),
            self._volume_intensity(primitives),
            self._level_proximity(context),
            self._volatility_regime(context),
        )
        total = round(sum(component.points for component in components))
        return FallbackScore(score=total, tier=classify_tier(total), components=components)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    def _delta_strength(self, primitives: Primitives) -> ScoreComponent:
        """20 points for a decisive delta in the analysed bar."""
        weight = self._weights["delta_strength"]
        ratio = abs(primitives.direction.delta.ratio)
        points = clamp(ratio / 0.6) * weight
        direction = primitives.direction.delta.direction
        return ScoreComponent("delta_strength", points, weight, f"delta ratio {ratio:.2f} ({direction})")

    def _orderbook_imbalance(self, primitives: Primitives) -> ScoreComponent:
        """20 points for a lopsided book."""
        weight = self._weights["orderbook_imbalance"]
        ratio = primitives.pressure.orderbook.imbalance_ratio
        # A ratio of 2.0 is treated as a fully decisive imbalance.
        points = clamp((ratio - 1.0) / 1.0) * weight
        return ScoreComponent(
            "orderbook_imbalance",
            points,
            weight,
            f"book {primitives.pressure.orderbook.imbalance_direction} {ratio:.2f}",
        )

    def _mtf_alignment(self, context: FallbackContext) -> ScoreComponent:
        """20 points for higher timeframe agreement."""
        weight = self._weights["mtf_alignment"]
        points = clamp(context.mtf_alignment) * weight
        return ScoreComponent(
            "mtf_alignment", points, weight, f"alignment {context.mtf_alignment:.2f}"
        )

    def _volume_intensity(self, primitives: Primitives) -> ScoreComponent:
        """15 points for volume that stands out from its baseline."""
        weight = self._weights["volume_intensity"]
        intensity = primitives.quality.volume_intensity
        # Reward both ends: unusually heavy volume confirms, unusually light
        # volume denies.
        if intensity.ratio >= 1.0:
            points = clamp((intensity.ratio - 1.0) / 1.5) * weight
            detail = f"volume {intensity.ratio:.2f}x average"
        else:
            points = clamp((1.0 - intensity.ratio) / 0.6) * weight
            detail = f"volume {intensity.ratio:.2f}x average (thin)"
        return ScoreComponent("volume_intensity", points, weight, detail)

    def _level_proximity(self, context: FallbackContext) -> ScoreComponent:
        """15 points for price sitting on a key level."""
        weight = self._weights["level_proximity"]
        points = clamp(context.level_proximity) * weight
        return ScoreComponent(
            "level_proximity", points, weight, f"proximity {context.level_proximity:.2f}"
        )

    def _volatility_regime(self, context: FallbackContext) -> ScoreComponent:
        """10 points for a tradable, non-pathological regime."""
        weight = self._weights["volatility_regime"]
        factor = VOLATILITY_POINTS.get(context.volatility_regime, VOLATILITY_DEFAULT)
        return ScoreComponent(
            "volatility_regime", factor * weight, weight, f"{context.volatility_regime} volatility"
        )