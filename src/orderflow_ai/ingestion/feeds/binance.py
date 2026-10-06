"""Binance websocket market data feeds.

The ``websockets`` dependency is imported lazily so that offline modes (replay,
synthetic) and the test suite never require it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ...core.config import ExchangeConfig
from ...core.models import BookSnapshot, DepthUpdate, Trade
from ..normalizer import BinanceNormalizer
from ..order_book import OrderBook
from .base import DepthFeed, TradeFeed

__all__ = ["BinanceDepthFeed", "BinanceTradeFeed"]

logger = logging.getLogger(__name__)

DEFAULT_WS_URL = "wss://stream.binance.com:9443"


class BinanceTradeFeed(TradeFeed):
    """Streams aggregated trades from ``/ws/<symbol>@aggTrade``."""

    name = "binance-trades"

    def __init__(self, exchange: ExchangeConfig, stream_name: str = "aggTrade") -> None:
        """Create a trade feed.

        Args:
            exchange: Exchange settings supplying the symbol and endpoints.
            stream_name: Binance stream suffix, for example ``aggTrade`` or
                ``trade``.
        """
        self._stream_name = stream_name
        self._normalizer = BinanceNormalizer(exchange.symbol, exchange.price_precision)
        self._url = _ws_url(exchange, stream_name)

    @property
    def url(self) -> str:
        """Websocket endpoint this feed connects to."""
        return self._url

    async def stream(self) -> AsyncIterator[Trade]:
        """Yield normalised trades, reconnecting on connection loss."""
        async for message in _json_messages(self._url):
            if message.get("e") not in {"aggTrade", "trade", None}:
                continue
            trade = self._normalizer.trade(message)
            if trade is not None:
                yield trade


class BinanceDepthFeed(DepthFeed):
    """Streams order book updates and maintains the local book state.

    The partial book stream (``depth20@100ms``) is the default because it is
    self-contained: every message carries a complete ladder for the top levels,
    so no REST bootstrap or diff resynchronisation is required.
    """

    name = "binance-depth"

    def __init__(self, exchange: ExchangeConfig, stream_name: str = "depth20@100ms") -> None:
        """Create a depth feed.

        Args:
            exchange: Exchange settings supplying the symbol and endpoints.
            stream_name: Binance depth stream suffix. Partial book streams carry
                full ladders, diff streams (``depth@100ms``) carry changes.
        """
        self._stream_name = stream_name
        self._normalizer = BinanceNormalizer(exchange.symbol, exchange.price_precision)
        self._url = _ws_url(exchange, stream_name)
        self._book = OrderBook(exchange.depth_levels)

    @property
    def url(self) -> str:
        """Websocket endpoint this feed connects to."""
        return self._url

    async def stream(self) -> AsyncIterator[DepthUpdate]:
        """Yield normalised depth updates and keep the local book in sync."""
        async for message in _json_messages(self._url):
            if message.get("e") not in {"depthUpdate", None}:
                continue
            update = self._normalizer.depth(message)
            if update is not None:
                self._book.apply(update)
                yield update

    def snapshot(self) -> BookSnapshot:
        """Return the book as reconstructed from the stream so far."""
        return self._book.snapshot()


def _ws_url(exchange: ExchangeConfig, stream_name: str) -> str:
    """Build the websocket endpoint for one symbol stream."""
    base = (exchange.ws_url or DEFAULT_WS_URL).rstrip("/")
    return f"{base}/ws/{exchange.symbol.lower()}@{stream_name}"


async def _json_messages(
    url: str,
    reconnect: bool = True,
    backoff_seconds: float = 0.5,
    max_backoff_seconds: float = 30.0,
) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON objects from a websocket endpoint, reconnecting on failure.

    Args:
        url: Endpoint to connect to.
        reconnect: Reconnect after a dropped connection instead of raising.
        backoff_seconds: Initial delay before the first reconnect attempt.
        max_backoff_seconds: Upper bound of the exponential backoff.

    Yields:
        One decoded JSON object per websocket message.
    """
    import websockets  # Imported lazily: only live mode needs websockets.

    delay = backoff_seconds
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as socket:
                delay = backoff_seconds
                async for raw in socket:
                    message = _decode(raw)
                    if message is not None:
                        yield message
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a feed must survive transport errors
            if not reconnect:
                raise
            logger.warning("websocket %s failed (%s); retrying in %.1fs", url, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_backoff_seconds)


def _decode(raw: str | bytes) -> dict[str, Any] | None:
    """Decode one websocket message, ignoring non-object payloads."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    # Combined stream envelopes nest the event under "data".
    nested = payload.get("data")
    return nested if isinstance(nested, dict) else payload