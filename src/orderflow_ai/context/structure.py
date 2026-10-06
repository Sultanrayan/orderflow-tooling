"""Swing detection and market structure labelling.

Swings are fractal pivots: a bar whose high (or low) is the extreme of the two
bars before and the two bars after it. Fractals need three closed bars on each
side, so the analysis always runs on closed bars only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..core.models import Candle
from .models import (
    BOS_BEARISH,
    BOS_BULLISH,
    BOS_NONE,
    STRUCTURE_DOWNTREND,
    STRUCTURE_RANGE,
    STRUCTURE_UNKNOWN,
    STRUCTURE_UPTREND,
    SWING_HH,
    SWING_HL,
    SWING_LH,
    SWING_LL,
    MarketStructure,
)

__all__ = ["Swing", "build_structure", "detect_swings", "label_swings"]

FRACTAL_STRENGTH = 2


@dataclass(frozen=True, slots=True)
class Swing:
    """A confirmed pivot in the price series."""

    price: float
    index: int
    is_high: bool
    label: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialisable view used by reports."""
        return {"price": self.price, "index": self.index, "high": self.is_high, "label": self.label}


def detect_swings(
    candles: Sequence[Candle], strength: int = FRACTAL_STRENGTH
) -> tuple[Swing, ...]:
    """Find confirmed swing highs and lows.

    Args:
        candles: Closed candles, oldest first.
        strength: Number of bars required on each side of the pivot.

    Returns:
        Swings ordered by position in the series.
    """
    if strength < 1 or len(candles) < 2 * strength + 1:
        return ()

    swings: list[Swing] = []
    for index in range(strength, len(candles) - strength):
        window = candles[index - strength : index + strength + 1]
        centre = window[strength]
        highs = [candle.high for candle in window]
        lows = [candle.low for candle in window]
        if centre.high == max(highs) and highs.count(centre.high) == 1:
            swings.append(Swing(price=centre.high, index=index, is_high=True))
        elif centre.low == min(lows) and lows.count(centre.low) == 1:
            swings.append(Swing(price=centre.low, index=index, is_high=False))
    return tuple(swings)


def label_swings(swings: Sequence[Swing]) -> tuple[Swing, ...]:
    """Tag each swing with HH, HL, LH or LL by comparing to the same kind."""
    highs = [swing for swing in swings if swing.is_high]
    lows = [swing for swing in swings if not swing.is_high]

    labelled: list[Swing] = []
    for index, swing in enumerate(highs):
        previous = highs[index - 1].price if index else None
        label = "" if previous is None else SWING_HH if swing.price > previous else SWING_LH
        labelled.append(Swing(swing.price, swing.index, True, label))
    for index, swing in enumerate(lows):
        previous = lows[index - 1].price if index else None
        label = "" if previous is None else SWING_HL if swing.price > previous else SWING_LL
        labelled.append(Swing(swing.price, swing.index, False, label))
    return tuple(sorted(labelled, key=lambda swing: swing.index))


def build_structure(candles: Sequence[Candle], strength: int = FRACTAL_STRENGTH) -> MarketStructure:
    """Describe the market structure behind a series of closed candles.

    Args:
        candles: Closed candles, oldest first.
        strength: Fractal strength for swing detection.

    Returns:
        The structure, including the most recent break of structure and whether
        it was a change of character.
    """
    swings = label_swings(detect_swings(candles, strength))
    highs = [swing.price for swing in swings if swing.is_high]
    lows = [swing.price for swing in swings if not swing.is_high]
    if not highs or not lows:
        return MarketStructure(label=STRUCTURE_UNKNOWN, bos=BOS_NONE)

    last_close = candles[-1].close
    bos = _last_break_of_structure(candles, highs[-1], lows[-1])
    label = _structure_label(highs, lows, last_close)
    choch = _is_change_of_character(highs, lows)

    return MarketStructure(
        label=label,
        bos=bos,
        choch=choch,
        last_swing_high=highs[-1],
        last_swing_low=lows[-1],
        swing_highs=tuple(highs[-3:]),
        swing_lows=tuple(lows[-3:]),
    )


def _structure_label(highs: Sequence[float], lows: Sequence[float], last_close: float) -> str:
    """Classify the trend as up, down or ranging."""
    if len(highs) < 2 or len(lows) < 2:
        return STRUCTURE_RANGE
    higher_highs = highs[-1] > highs[-2]
    higher_lows = lows[-1] > lows[-2]
    lower_highs = highs[-1] < highs[-2]
    lower_lows = lows[-1] < lows[-2]

    if higher_highs and higher_lows:
        return STRUCTURE_UPTREND
    if lower_highs and lower_lows:
        return STRUCTURE_DOWNTREND
    if last_close > highs[-1] or last_close < lows[-1]:
        return STRUCTURE_RANGE
    return STRUCTURE_RANGE


def _last_break_of_structure(
    candles: Sequence[Candle], swing_high: float, swing_low: float
) -> str:
    """Return the direction of the most recent close beyond a swing."""
    for candle in reversed(candles):
        if candle.close > swing_high:
            return BOS_BULLISH
        if candle.close < swing_low:
            return BOS_BEARISH
    return BOS_NONE


def _is_change_of_character(highs: Sequence[float], lows: Sequence[float]) -> bool:
    """``True`` when the last two highs and lows moved in opposite directions."""
    if len(highs) < 2 or len(lows) < 2:
        return False
    highs_flipped = highs[-1] < highs[-2]
    lows_flipped = lows[-1] > lows[-2]
    return highs_flipped and lows_flipped