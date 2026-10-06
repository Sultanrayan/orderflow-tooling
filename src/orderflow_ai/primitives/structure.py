"""Structure primitives: footprint, volume profile and value migration."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.models import FootprintBar
from .models import Footprint, FootprintRow, ValueMigration, VolumeProfile
from .statistics import clamp, mean, safe_div

__all__ = [
    "compute_footprint",
    "compute_value_migration",
    "compute_volume_profile",
    "point_of_control",
    "poc_series",
]

VALUE_AREA_FRACTION = 0.7


def point_of_control(bar: FootprintBar) -> float | None:
    """Price that absorbed the most volume inside a single bar."""
    if not bar.levels:
        return None
    busiest = max(bar.levels.values(), key=lambda level: level.total_volume)
    return busiest.price if busiest.total_volume > 0 else None


def poc_series(bars: Sequence[FootprintBar]) -> list[float]:
    """Points of control of ``bars``, skipping bars without volume."""
    controls = (point_of_control(bar) for bar in bars)
    return [price for price in controls if price is not None]


def compute_footprint(bar: FootprintBar, max_levels: int = 0) -> Footprint:
    """Project a footprint bar into a :class:`Footprint` value object.

    Args:
        bar: The closed bar to project.
        max_levels: Keep only the ``max_levels`` highest volume levels when
            greater than zero. Zero keeps every level.

    Returns:
        The footprint primitive for this bar.
    """
    rows = [FootprintRow.of(level) for level in bar.sorted_levels]
    if max_levels > 0 and len(rows) > max_levels:
        rows.sort(key=lambda row: row.total_volume, reverse=True)
        rows = sorted(rows[:max_levels], key=lambda row: row.price)

    return Footprint(
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        total_volume=bar.total_volume,
        buy_volume=bar.buy_volume,
        sell_volume=bar.sell_volume,
        delta=bar.delta,
        trade_count=bar.trade_count,
        levels=tuple(rows),
    )


def compute_volume_profile(
    bars: Sequence[FootprintBar], price_decimals: int = 6, node_limit: int = 3
) -> VolumeProfile:
    """Build a volume profile across the supplied bars.

    Volume is aggregated per price level. The value area is the contiguous set
    of levels around the point of control holding
    :data:`VALUE_AREA_FRACTION` of the traded volume. High and low volume nodes
    are the extremes of the distribution.

    Args:
        bars: Bars to aggregate, any order.
        price_decimals: Decimal places used when bucketing prices.
        node_limit: Maximum number of high/low volume nodes returned per side.

    Returns:
        The volume profile, or a degenerate profile at the mid price when no
        bars or no volume are available.
    """
    histogram: dict[float, float] = {}
    for bar in bars:
        for level in bar.levels.values():
            histogram[round(level.price, price_decimals)] = (
                histogram.get(round(level.price, price_decimals), 0.0) + level.total_volume
            )

    histogram = {price: volume for price, volume in histogram.items() if volume > 0}
    if not histogram:
        mid = bars[-1].close if bars else 0.0
        return VolumeProfile(poc=mid, vah=mid, val=mid)

    prices = sorted(histogram)
    poc = max(prices, key=lambda price: histogram[price])
    total = sum(histogram.values())
    target = total * VALUE_AREA_FRACTION

    # Grow the value area outwards from the point of control, always taking
    # the side with the larger volume next.
    poc_index = prices.index(poc)
    low_index = high_index = poc_index
    covered = histogram[poc]
    while covered < target and (low_index > 0 or high_index < len(prices) - 1):
        below = histogram[prices[low_index - 1]] if low_index > 0 else -1.0
        above = histogram[prices[high_index + 1]] if high_index < len(prices) - 1 else -1.0
        if below >= above:
            low_index -= 1
            covered += below
        else:
            high_index += 1
            covered += above
    val, vah = prices[low_index], prices[high_index]

    average = mean(list(histogram.values()))
    high_nodes = tuple(
        sorted(
            (price for price, volume in histogram.items() if volume >= average * 1.5),
            key=lambda price: histogram[price],
            reverse=True,
        )[:node_limit]
    )
    low_nodes = tuple(
        sorted(
            (price for price, volume in histogram.items() if volume <= average * 0.5)
        )[:node_limit]
    )

    return VolumeProfile(
        poc=poc,
        vah=vah,
        val=val,
        hvn=high_nodes,
        lvn=low_nodes,
        total_volume=total,
    )


def compute_value_migration(
    profile: VolumeProfile,
    previous_pocs: Sequence[float],
    tick_size: float,
    max_ticks: float = 10.0,
) -> ValueMigration:
    """Describe how the point of control has moved.

    Args:
        profile: Current volume profile.
        previous_pocs: Points of control of the preceding bars, oldest first.
        tick_size: Smallest tradable price increment.
        max_ticks: Tick distance at which migration strength saturates at 1.0.

    Returns:
        The value migration primitive. With no history the migration is flat.
    """
    if not previous_pocs:
        return ValueMigration(
            poc_direction="flat",
            migration_strength=0.0,
            poc_current=profile.poc,
            poc_previous=profile.poc,
            ticks=0.0,
        )

    previous = mean(previous_pocs)
    ticks = safe_div(profile.poc - previous, tick_size)
    direction = "up" if ticks > 0.5 else "down" if ticks < -0.5 else "flat"
    return ValueMigration(
        poc_direction=direction,
        migration_strength=clamp(abs(ticks) / max_ticks),
        poc_current=profile.poc,
        poc_previous=previous,
        ticks=ticks,
    )