"""Assembly of the ten primitives into one result object.

:class:`PrimitiveEngine` is the single entry point of layer 2. It owns no
state: give it the same bars and book and it always returns the same
:class:`~orderflow_ai.primitives.models.Primitives`.
"""

from __future__ import annotations

from ..core.config import PrimitivesConfig
from ..core.models import BookSnapshot, FootprintBar
from .direction import compute_cvd, compute_delta, compute_delta_divergence
from .models import (
    DirectionPrimitives,
    PressurePrimitives,
    PrimitiveInput,
    Primitives,
    QualityPrimitives,
    StructurePrimitives,
)
from .pressure import compute_imbalance, compute_orderbook_depth
from .quality import compute_absorption, compute_volume_intensity
from .structure import (
    compute_footprint,
    compute_value_migration,
    compute_volume_profile,
    poc_series,
)

__all__ = ["PrimitiveEngine"]

PROFILE_LOOKBACK = 10
CVD_LOOKBACK = 20
INTENSITY_LOOKBACK = 20


class PrimitiveEngine:
    """Computes all ten order flow primitives for a closed bar."""

    def __init__(
        self,
        config: PrimitivesConfig | None = None,
        tick_size: float = 0.01,
        depth_levels: int = 20,
        price_decimals: int = 6,
    ) -> None:
        """Create the engine.

        Args:
            config: Primitive thresholds. Defaults are used when omitted.
            tick_size: Smallest price increment, used to scale value migration.
            depth_levels: Book levels included in the depth primitive.
            price_decimals: Decimal places used when bucketing price levels.
        """
        self._config = config or PrimitivesConfig()
        self._tick_size = tick_size or 0.01
        self._depth_levels = depth_levels
        self._price_decimals = price_decimals

    @property
    def config(self) -> PrimitivesConfig:
        """Thresholds in use."""
        return self._config

    def compute(self, bar: FootprintBar, history: tuple[FootprintBar, ...] = (), book=None) -> Primitives:
        """Compute every primitive for ``bar``.

        Args:
            bar: The bar that just closed.
            history: Previously closed bars, oldest first.
            book: Order book state at the close of ``bar``.

        Returns:
            The grouped primitives for this bar.
        """
        data = PrimitiveInput(
            bar=bar,
            history=history,
            book=book if book is not None else BookSnapshot.empty(),
            config=self._config,
            price_decimals=self._price_decimals,
        )
        return self._compute(data)

    def _compute(self, data: PrimitiveInput) -> Primitives:
        """Run the individual primitive functions over ``data``."""
        bar, history = data.bar, data.history

        footprint = compute_footprint(bar)
        profile = compute_volume_profile(
            [*history[-(PROFILE_LOOKBACK - 1):], bar], data.price_decimals
        )
        migration = compute_value_migration(
            profile, poc_series(history[-(PROFILE_LOOKBACK - 1):]), self._tick_size
        )

        delta = compute_delta(bar, data.config.delta_ratio_threshold)
        cvd = compute_cvd(bar, history, CVD_LOOKBACK)
        divergence = compute_delta_divergence(bar, history)

        imbalance = compute_imbalance(bar, data.config.imbalance_ratio_threshold)
        orderbook = compute_orderbook_depth(data.book, self._depth_levels)

        intensity = compute_volume_intensity(bar, history, INTENSITY_LOOKBACK)
        absorption = compute_absorption(
            bar,
            history,
            data.config.absorption_min_confidence,
            data.config.delta_ratio_threshold,
            INTENSITY_LOOKBACK,
        )

        return Primitives(
            symbol=bar.symbol,
            timeframe=bar.timeframe,
            open_time=bar.open_time,
            structure=StructurePrimitives(
                footprint=footprint, volume_profile=profile, value_migration=migration
            ),
            direction=DirectionPrimitives(delta=delta, cvd=cvd, delta_divergence=divergence),
            pressure=PressurePrimitives(imbalance=imbalance, orderbook=orderbook),
            quality=QualityPrimitives(volume_intensity=intensity, absorption=absorption),
        )