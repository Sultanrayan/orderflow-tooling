"""Exchange payload normalisation.

Each supported exchange delivers raw dictionaries with its own field names and
its own interpretation of "side". :class:`BinanceNormalizer` converts those
payloads into the canonical models from :mod:`orderflow_ai.core.models` so no
other layer needs to know where a tick came from.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

from ..core.models import BookSnapshot, Candle, DepthLevel, DepthUpdate, Side, Trade
from ..core.timeframes import Timeframe

__all__ = ["BinanceNormalizer", "Normalizer"]


class Normalizer:
    """Interface implemented by every exchange normaliser."""

    name = "generic"

    def trade(self, payload: dict[str, Any]) -> Trade | None:
        """Convert a raw trade event into a :class:`Trade`, or ``None`` to skip."""
        raise NotImplementedError

    def depth(self, payload: dict[str, Any]) -> DepthUpdate | None:
        """Convert a raw depth event into a :class:`DepthUpdate`."""
        raise NotImplementedError

    def book(self, payload: dict[str, Any]) -> BookSnapshot:
        """Convert a raw partial book event into a :class:`BookSnapshot`."""
        raise NotImplementedError

    def candle(self, payload: Any, timeframe: Timeframe) -> Candle:
        """Convert a raw kline payload into a :class:`Candle`."""
        raise NotImplementedError


class BinanceNormalizer(Normalizer):
    """Normalises Binance spot market data streams.

    Handles ``aggTrade``/``trade`` streams, both the partial book stream
    (``depth20@100ms``) and the diff stream (``depth@100ms``), plus REST
    klines.
    """

    name = "binance"

    def __init__(self, symbol: str = "BTCUSDT", price_decimals: int = 6) -> None:
        self.symbol = symbol.upper()
        self.price_decimals = price_decimals

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def trade(self, payload: dict[str, Any]) -> Trade | None:
        """Normalise an aggregated or raw trade event.

        Binance reports the *maker* side in ``m``. The aggressor is therefore
        ``SELL`` when ``m`` is true and ``BUY`` otherwise.
        """
        price = _float(payload.get("p"))
        size = _float(payload.get("q"))
        if price <= 0 or size <= 0:
            return None

        raw_ts = payload.get("T") or payload.get("E")
        return Trade(
            symbol=str(payload.get("s") or self.symbol).upper(),
            ts=_float(raw_ts) / 1000.0,
            price=price,
            size=size,
            side=self._aggressor(payload),
            source=self.name,
        )

    @staticmethod
    def _aggressor(payload: dict[str, Any]) -> Side:
        if "m" in payload:
            return Side.SELL if bool(payload["m"]) else Side.BUY
        return Side.BUY if str(payload.get("side", "BUY")).upper().startswith("B") else Side.SELL

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------
    def depth(self, payload: dict[str, Any]) -> DepthUpdate | None:
        """Normalise a depth diff event into canonical levels."""
        event_time = _float(payload.get("E") or payload.get("T"))
        if "b" in payload or "a" in payload:
            return DepthUpdate(
                ts=event_time / 1000.0,
                bids=_levels(payload.get("b")),
                asks=_levels(payload.get("a")),
                is_snapshot=False,
            )
        # Partial book stream (``depth20@100ms``) carries whole ladders.
        return DepthUpdate(
            ts=event_time / 1000.0,
            bids=_levels(payload.get("bids")),
            asks=_levels(payload.get("asks")),
            is_snapshot=True,
        )

    def book(self, payload: dict[str, Any]) -> BookSnapshot:
        """Normalise a partial book event into a snapshot."""
        return BookSnapshot(bids=_levels(payload.get("bids")), asks=_levels(payload.get("asks")))

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------
    def candle(self, payload: Any, timeframe: Timeframe) -> Candle:
        """Normalise a REST kline row (or a websocket event) into a candle.

        Binance REST returns ``[open_time, open, high, low, close, volume,
        close_time, ...]`` while the websocket kline event nests the same
        fields under ``k``.

        Args:
            payload: A REST kline row or a websocket kline event.
            timeframe: Timeframe of the kline. Kept for interface symmetry with
                other normalisers; Binance encodes it in the payload already.

        Returns:
            The normalised candle.
        """
        if isinstance(payload, Mapping):
            fields = payload.get("k", payload)
            return Candle(
                open_time=_to_utc(fields.get("t")),
                open=_float(fields.get("o")),
                high=_float(fields.get("h")),
                low=_float(fields.get("l")),
                close=_float(fields.get("c")),
                volume=_float(fields.get("v")),
                close_time=_to_utc(fields.get("T")),
                trade_count=int(_float(fields.get("n"))),
                closed=True,
            )

        return Candle(
            open_time=_to_utc(payload[0]),
            open=_float(payload[1]),
            high=_float(payload[2]),
            low=_float(payload[3]),
            close=_float(payload[4]),
            volume=_float(payload[5]),
            close_time=_to_utc(payload[6]),
            trade_count=int(_float(payload[8])),
            closed=True,
        )


def _to_utc(epoch_ms: Any) -> dt.datetime:
    """Convert an exchange millisecond timestamp into an aware UTC datetime."""
    return dt.datetime.fromtimestamp(_float(epoch_ms) / 1000.0, tz=dt.UTC)


def _float(value: Any) -> float:
    """Parse a value that an exchange may send as a string or a number."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _levels(rows: Any) -> tuple[DepthLevel, ...]:
    """Convert ``[["price", "size"], ...]`` rows into :class:`DepthLevel` items."""
    if not rows:
        return ()
    levels: list[DepthLevel] = []
    for row in rows:
        price, size = _float(row[0]), _float(row[1])
        if price > 0 and size > 0:
            levels.append(DepthLevel(price=price, size=size))
    return tuple(levels)