"""Pressure primitives: footprint imbalance and live order book depth."""

from __future__ import annotations

from ..core.models import BookSnapshot, FootprintBar, Side
from .models import Imbalance, ImbalanceLevel, OrderBookDepth, Wall
from .statistics import clamp, mean, safe_div

__all__ = ["compute_imbalance", "compute_orderbook_depth"]

WALL_SIZE_FACTOR = 3.0
MIN_WALL_SIZE = 5.0


def compute_imbalance(bar: FootprintBar, threshold: float = 1.5) -> Imbalance:
    """Compute diagonal imbalances of a footprint bar.

    A level is imbalanced when one side traded ``threshold`` times more than the
    other. Ratios are reported for every two sided level so the payload can
    always show the strongest reading even below the threshold.

    Args:
        bar: The closed bar to analyse.
        threshold: Ratio above which a level counts as imbalanced.

    Returns:
        The imbalance primitive with the qualifying levels and the strongest of
        all two sided levels.
    """
    candidates: list[ImbalanceLevel] = []
    imbalanced: list[ImbalanceLevel] = []

    for level in bar.sorted_levels:
        if level.bid_volume <= 0 or level.ask_volume <= 0:
            continue
        ratio = level.imbalance_ratio
        side = level.imbalance_side or Side.BUY
        entry = ImbalanceLevel(
            price=level.price,
            ratio=ratio,
            side=side,
            bid_volume=level.bid_volume,
            ask_volume=level.ask_volume,
        )
        candidates.append(entry)
        if ratio >= threshold:
            imbalanced.append(entry)

    strongest = max(candidates, key=lambda item: item.ratio, default=None)
    imbalanced.sort(key=lambda item: item.ratio, reverse=True)
    return Imbalance(threshold=threshold, levels=tuple(imbalanced), strongest=strongest)


def compute_orderbook_depth(
    book: BookSnapshot,
    levels: int = 20,
    wall_factor: float = WALL_SIZE_FACTOR,
    min_wall_size: float = MIN_WALL_SIZE,
) -> OrderBookDepth:
    """Compute live book metrics for the top ``levels`` prices.

    Args:
        book: Current book snapshot.
        levels: Number of levels per side to include.
        wall_factor: Multiple of the average level size that marks a wall.
        min_wall_size: Absolute size a wall must also exceed.

    Returns:
        The order book depth primitive. ``imbalance_direction`` is ``bid``,
        ``ask`` or ``balanced``.
    """
    bid_depth = book.bid_depth(levels)
    ask_depth = book.ask_depth(levels)
    ratio = safe_div(bid_depth, ask_depth, default=1.0)
    if ratio > 1.05:
        direction = "bid"
    elif ratio < 0.95:
        direction = "ask"
    else:
        direction = "balanced"

    return OrderBookDepth(
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        imbalance_ratio=ratio,
        imbalance_direction=direction,
        spread_bps=book.spread_bps,
        walls=_find_walls(book, levels, wall_factor, min_wall_size),
    )


def _find_walls(
    book: BookSnapshot, levels: int, wall_factor: float, min_wall_size: float
) -> tuple[Wall, ...]:
    """Locate resting orders that dwarf their neighbours."""
    ladders = (book.bids[:levels], book.asks[:levels])
    sizes = [level.size for side in ladders for level in side]
    average = mean(sizes)
    if average <= 0:
        return ()

    walls: list[Wall] = []
    for side_name, ladder in zip((Side.BUY, Side.SELL), ladders, strict=True):
        for level in ladder:
            if level.size < min_wall_size:
                continue
            strength = level.size / average
            if strength >= wall_factor:
                walls.append(
                    Wall(price=level.price, side=side_name, size=level.size, strength=clamp(strength / 10))
                )
    walls.sort(key=lambda wall: wall.strength, reverse=True)
    return tuple(walls[:3])