"""REST candle feed.

Historical bars come from the exchange REST API rather than the websocket
market data stream: the higher timeframe context only needs the last few closed
candles, and polling keeps the websocket connections dedicated to ticks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from ...core.config import ExchangeConfig
from ...core.models import Candle
from ...core.timeframes import Timeframe, floor_timestamp
from ..normalizer import BinanceNormalizer

__all__ = ["OHLCVFeed"]

logger = logging.getLogger(__name__)

DEFAULT_REST_URL = "https://api.binance.com"


class OHLCVFeed:
    """Polls closed candles for one symbol and timeframe."""

    name = "ohlcv"

    def __init__(
        self,
        exchange: ExchangeConfig,
        timeframe: Timeframe,
        limit: int = 200,
        poll_seconds: float | None = None,
    ) -> None:
        """Create the feed.

        Args:
            exchange: Exchange settings supplying the symbol and REST endpoint.
            timeframe: Candle timeframe to fetch.
            limit: Number of candles requested per poll.
            poll_seconds: Fixed poll interval. Defaults to waiting for the next
                candle boundary so the API is queried once per closed bar.
        """
        self._exchange = exchange
        self._timeframe = timeframe
        self._limit = limit
        self._poll_seconds = poll_seconds
        self._normalizer = BinanceNormalizer(exchange.symbol, exchange.price_precision)
        self._emitted: float = 0.0

    @property
    def url(self) -> str:
        """REST endpoint used for klines."""
        base = (self._exchange.base_url or DEFAULT_REST_URL).rstrip("/")
        return f"{base}/api/v3/klines"

    async def history(self, limit: int | None = None) -> list[Candle]:
        """Fetch recent closed candles, oldest first.

        Args:
            limit: Override for the number of candles.

        Returns:
            Closed candles ordered from oldest to newest.
        """
        import httpx  # Imported lazily: only live mode needs an HTTP client.

        params = {
            "symbol": self._exchange.symbol.upper(),
            "interval": self._timeframe.value,
            "limit": limit or self._limit,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.url, params=params)
            response.raise_for_status()
            rows = response.json()

        return [self._normalizer.candle(row, self._timeframe) for row in rows]

    async def stream(self) -> AsyncIterator[Candle]:
        """Yield the recent history, then every newly closed candle."""
        for candle in await self.history():
            self._emitted = max(self._emitted, candle.close_time.timestamp())
            yield candle

        while True:
            await asyncio.sleep(self._seconds_until_next_poll())
            try:
                candles = await self.history(limit=2)
            except Exception as exc:  # noqa: BLE001 - keep polling through API errors
                logger.warning("OHLCV poll failed for %s: %s", self._exchange.symbol, exc)
                continue
            for candle in candles:
                if candle.close_time.timestamp() > self._emitted:
                    self._emitted = candle.close_time.timestamp()
                    yield candle

    def _seconds_until_next_poll(self) -> float:
        """Seconds to wait before the next poll."""
        if self._poll_seconds is not None:
            return self._poll_seconds
        now = time.time()
        next_boundary = floor_timestamp(now, self._timeframe) + self._timeframe.seconds
        return max(1.0, next_boundary - now + 0.25)