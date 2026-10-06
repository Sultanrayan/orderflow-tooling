"""Direction primitives: delta, cumulative delta and delta divergence."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import FootprintBar
from .models import BIAS_BEARISH, BIAS_BULLISH, BIAS_NEUTRAL, Cvd, Delta, DeltaDivergence
from .statistics import clamp, linear_slope, mean, safe_div

__all__ = ["compute_cvd", "compute_delta", "compute_delta_divergence"]

MOMENTUM_ACCELERATING = "accelerating"
MOMENTUM_BUILDING = "building"
MOMENTUM_NEUTRAL = "neutral"
MOMENTUM_FADING = "fading"

DIVERGENCE_BEARISH = "bearish"
DIVERGENCE_BULLISH = "bullish"


def compute_delta(bar: FootprintBar, ratio_threshold: float = 0.15) -> Delta:
    """Compute the aggressive volume balance of ``bar``.

    Args:
        bar: The closed bar to analyse.
        ratio_threshold: Absolute delta ratio above which the bias counts as
            directional.

    Returns:
        The delta primitive. ``direction`` is ``bullish`` or ``bearish`` when
        the ratio exceeds ``ratio_threshold``, otherwise ``neutral``.
    """
    total = bar.total_volume
    value = bar.delta
    ratio = safe_div(value, total)
    if ratio > ratio_threshold:
        direction = BIAS_BULLISH
    elif ratio < -ratio_threshold:
        direction = BIAS_BEARISH
    else:
        direction = BIAS_NEUTRAL

    return Delta(
        value=value,
        ratio=ratio,
        direction=direction,
        strength=clamp(abs(ratio)),
        buy_volume=bar.buy_volume,
        sell_volume=bar.sell_volume,
        total_volume=total,
    )


def compute_cvd(
    bar: FootprintBar, history: Sequence[FootprintBar], lookback: int = 20
) -> Cvd:
    """Compute cumulative volume delta over the lookback window.

    ``momentum`` compares the slope of the delta series with its own scale:
    a slope above one average bar delta means the buyers are accelerating.

    Args:
        bar: The most recent closed bar.
        history: Previously closed bars, oldest first.
        lookback: Maximum number of bars to include.

    Returns:
        The CVD primitive.
    """
    bars = [*history[-lookback:], bar]
    deltas = [candidate.delta for candidate in bars]
    value = sum(deltas)
    slope = linear_slope(deltas)
    baseline = mean([abs(candidate.delta) for candidate in bars]) or 1.0
    normalised_slope = safe_div(slope, baseline, default=0.0)

    if normalised_slope > 0.75:
        momentum = MOMENTUM_ACCELERATING
    elif normalised_slope > 0.15:
        momentum = MOMENTUM_BUILDING
    elif normalised_slope < -0.75:
        momentum = MOMENTUM_FADING
    else:
        momentum = MOMENTUM_NEUTRAL

    return Cvd(value=value, slope=normalised_slope, momentum=momentum, bars=len(bars))


def compute_delta_divergence(
    bar: FootprintBar,
    history: Sequence[FootprintBar],
    min_bars: int = 3,
    lookback: int = 10,
) -> DeltaDivergence:
    """Detect price and delta moving in opposite directions.

    Args:
        bar: The most recent closed bar.
        history: Previously closed bars, oldest first.
        min_bars: Bars required before a divergence can be declared.
        lookback: Maximum number of bars to inspect.

    Returns:
        The divergence primitive. ``type`` is ``bearish`` when price rose while
        delta fell, and ``bullish`` for the mirror case.
    """
    bars = [*history[-(lookback - 1):], bar] if lookback > 1 else [bar]
    if len(bars) < min_bars:
        return DeltaDivergence(detected=False, bars_observed=len(bars))

    start_price = bars[0].open
    price_change = safe_div(bar.close - start_price, start_price)
    delta_change = sum(candidate.delta for candidate in bars)

    price_up, price_down = price_change > 0.001, price_change < -0.001
    delta_up, delta_down = delta_change > 0, delta_change < 0
    if not (price_up and delta_down or price_down and delta_up):
        return DeltaDivergence(
            detected=False,
            bars_observed=len(bars),
            price_change=price_change,
            delta_change=delta_change,
        )

    # Confidence grows with the number of confirming bars and with how decisive
    # the price move was relative to the typical bar range.
    typical_range = mean([candidate.range for candidate in bars]) or 1e-9
    range_ratio = clamp(abs(bar.close - start_price) / typical_range)
    depth = clamp((len(bars) - min_bars + 1) / max(1, lookback - min_bars + 1))
    confidence = round(100 * (0.6 * depth + 0.4 * range_ratio))

    return DeltaDivergence(
        detected=True,
        type=DIVERGENCE_BEARISH if price_up else DIVERGENCE_BULLISH,
        confidence=confidence,
        bars_observed=len(bars),
        price_change=price_change,
        delta_change=delta_change,
    )