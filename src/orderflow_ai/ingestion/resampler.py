"""Resampling of base timeframe bars into longer timeframes.

Layer 4 needs higher timeframe context. In live mode that history comes from the
exchange REST API; offline (replay and synthetic runs) it is derived from the
base bars the pipeline already produced, which keeps every mode consistent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from ..core.models import Candle, FootprintBar
from ..core.timeframes import Timeframe, bar_open_time

__all__ = ["resample_bars"]


def resample_bars(bars: Sequence[FootprintBar], target: Timeframe) -> list[Candle]:
    """Aggregate footprint bars into candles of ``target`` timeframe.

    Args:
        bars: Base timeframe bars, any order; they are sorted by open time.
        target: Target timeframe. It must be at least as long as the base.

    Returns:
        Completed target timeframe candles, oldest first. The final group is
        included only when it holds more than one base bar, because a single
        base bar is still forming.
    """
    if not bars:
        return []

    ordered = sorted(bars, key=lambda bar: bar.open_time)
    groups: dict[float, list[FootprintBar]] = {}
    for bar in ordered:
        open_ts = bar.open_time.timestamp()
        key = bar_open_time(open_ts, target).timestamp()
        groups.setdefault(key, []).append(bar)

    candles: list[Candle] = []
    for open_ts in sorted(groups):
        members = groups[open_ts]
        if len(members) < 2:
            continue
        first, last = members[0], members[-1]
        candles.append(
            Candle(
                open_time=dt.datetime.fromtimestamp(open_ts, tz=dt.UTC),
                open=first.open,
                high=max(bar.high for bar in members),
                low=min(bar.low for bar in members),
                close=last.close,
                volume=sum(bar.total_volume for bar in members),
                close_time=last.open_time + dt.timedelta(seconds=target.seconds),
                trade_count=sum(bar.trade_count for bar in members),
                closed=True,
            )
        )
    return candles