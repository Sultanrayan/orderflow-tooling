"""Key level construction.

Levels are ranked by how often price reacted there: confirmed swing points
first, then the point of control and the value area edges, then round numbers.
The pipeline only ships the closest few, so the ranking matters.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..primitives.models import VolumeProfile
from .models import KeyLevels
from .structure import Swing

__all__ = ["build_key_levels", "round_numbers"]

ROUND_STEP = (10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01)


def build_key_levels(
    price: float,
    profile: VolumeProfile,
    swings: Sequence[Swing] = (),
    max_levels: int = 3,
    tolerance: float = 0.0,
) -> KeyLevels:
    """Assemble support and resistance around the current price.

    Args:
        price: Current price, used to split levels into support and resistance.
        profile: Volume profile supplying POC, value area edges and nodes.
        swings: Detected swing points.
        max_levels: Levels kept per side, nearest first.
        tolerance: Prices closer than this are merged into one level.

    Returns:
        The ranked key levels.
    """
    swing_highs = [swing.price for swing in swings if swing.is_high]
    swing_lows = [swing.price for swing in swings if not swing.is_high]

    resistance = _dedupe([*swing_highs, profile.vah, *profile.hvn], tolerance)
    support = _dedupe([*swing_lows, profile.val, *profile.lvn], tolerance)

    resistance = [level for level in resistance if level > price]
    support = [level for level in support if level < price]

    return KeyLevels(
        support=tuple(sorted(support, reverse=True)[:max_levels]),
        resistance=tuple(sorted(resistance)[:max_levels]),
        poc=profile.poc,
        value_area_high=profile.vah,
        value_area_low=profile.val,
    )


def round_numbers(price: float, count: int = 2) -> tuple[float, ...]:
    """Round numbers bracketing ``price`` at the coarsest sensible step.

    Args:
        price: Reference price.
        count: How many levels per side to return.

    Returns:
        Up to ``2 * count`` round numbers, nearest first on each side.
    """
    levels: list[float] = []
    for step in ROUND_STEP:
        centre = round(price / step) * step
        for offset in (0, 1, -1, 2, -2):
            level = round(centre + offset * step, 10)
            if level > 0:
                levels.append(level)
        if len(levels) >= 4 * count:
            break

    above = sorted({level for level in levels if level > price})[:count]
    below = sorted({level for level in levels if level < price}, reverse=True)[:count]
    return tuple(above + below)


def _dedupe(levels: Sequence[float], tolerance: float) -> list[float]:
    """Sort levels and merge the ones that sit within ``tolerance``."""
    ordered = sorted({round(level, 10) for level in levels if level > 0})
    if tolerance <= 0:
        return ordered

    merged: list[float] = []
    for level in ordered:
        if merged and abs(level - merged[-1]) <= tolerance:
            continue
        merged.append(level)
    return merged