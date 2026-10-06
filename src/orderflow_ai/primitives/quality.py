"""Quality primitives: volume intensity and absorption score."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import FootprintBar, Side
from .models import AbsorptionScore, VolumeIntensity
from .statistics import clamp, mean, percentile_rank, safe_div
from .structure import point_of_control

__all__ = ["compute_absorption", "compute_volume_intensity"]

INTENSITY_LOW = "low"
INTENSITY_NORMAL = "normal"
INTENSITY_HIGH = "high"
INTENSITY_EXTREME = "extreme"

ABSORPTION_LEVEL_SHARE = 0.4


def compute_volume_intensity(
    bar: FootprintBar, history: Sequence[FootprintBar], lookback: int = 20
) -> VolumeIntensity:
    """Compare the bar's volume with its own recent baseline.

    Args:
        bar: The closed bar to analyse.
        history: Previously closed bars, oldest first.
        lookback: Number of bars forming the baseline.

    Returns:
        The volume intensity primitive. With no history the bar is reported as
        ``normal`` with a ratio of ``1.0``.
    """
    baseline = [candidate.total_volume for candidate in history[-lookback:] if candidate.total_volume > 0]
    if not baseline:
        return VolumeIntensity(
            ratio=1.0,
            percentile=50.0,
            intensity=INTENSITY_NORMAL,
            volume=bar.total_volume,
            average_volume=bar.total_volume,
        )

    average = mean(baseline)
    ratio = safe_div(bar.total_volume, average, default=1.0)
    percentile = percentile_rank(bar.total_volume, [*baseline, bar.total_volume]) * 100

    if ratio >= 2.5:
        intensity = INTENSITY_EXTREME
    elif ratio >= 1.5:
        intensity = INTENSITY_HIGH
    elif ratio <= 0.7:
        intensity = INTENSITY_LOW
    else:
        intensity = INTENSITY_NORMAL

    return VolumeIntensity(
        ratio=ratio,
        percentile=percentile,
        intensity=intensity,
        volume=bar.total_volume,
        average_volume=average,
    )


def compute_absorption(
    bar: FootprintBar,
    history: Sequence[FootprintBar],
    min_confidence: float = 60.0,
    delta_ratio_threshold: float = 0.15,
    lookback: int = 20,
) -> AbsorptionScore:
    """Score how much volume was absorbed without price progress.

    Four independent readings contribute to the score, each capped so that no
    single factor can carry it alone:

    * heavy volume against the recent baseline,
    * a bar range that is small for that volume,
    * a decisive delta showing who was aggressive,
    * repeated fills concentrated on one level (the classic footprint tell).

    Args:
        bar: The closed bar to analyse.
        history: Previously closed bars, oldest first.
        min_confidence: Score at which absorption is reported as detected.
        delta_ratio_threshold: Delta ratio considered decisive.
        lookback: Number of bars forming the baseline.

    Returns:
        The absorption primitive. A bar without volume scores zero: there is
        nothing to absorb.
    """
    if bar.total_volume <= 0 or not bar.levels:
        return AbsorptionScore(detected=False, confidence=0.0, price=bar.close)

    baseline = list(history[-lookback:])
    average_volume = mean([candidate.total_volume for candidate in baseline]) or bar.total_volume
    average_range = mean([candidate.range for candidate in baseline]) or bar.range

    volume_ratio = safe_div(bar.total_volume, average_volume, default=1.0)
    range_ratio = safe_div(bar.range, average_range, default=1.0)
    delta_ratio = safe_div(bar.delta, bar.total_volume)

    volume_points = min(30.0, max(0.0, (volume_ratio - 1.0) * 20.0))
    compression_points = min(25.0, max(0.0, (1.0 - range_ratio) * 25.0))
    delta_points = min(25.0, clamp(abs(delta_ratio) / max(delta_ratio_threshold, 1e-9)) * 25.0)

    concentration = _max_level_share(bar)
    cluster_points = 25.0 if concentration >= ABSORPTION_LEVEL_SHARE else 0.0

    confidence = round(clamp((volume_points + compression_points + delta_points + cluster_points) / 100.0) * 100)
    side = Side.BUY if bar.delta > 0 else Side.SELL if bar.delta < 0 else None

    return AbsorptionScore(
        detected=confidence >= min_confidence,
        confidence=confidence,
        side=side,
        price=point_of_control(bar) or bar.close,
        volume_ratio=volume_ratio,
        range_ratio=range_ratio,
    )


def _max_level_share(bar: FootprintBar) -> float:
    """Share of the bar's volume traded on its busiest level."""
    total = bar.total_volume
    if total <= 0 or not bar.levels:
        return 0.0
    busiest = max(level.total_volume for level in bar.levels.values())
    return busiest / total