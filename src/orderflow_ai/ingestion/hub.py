"""Feed orchestration for the pipeline.

:class:`IngestionHub` owns the feeds, the local order book, the higher timeframe
candle cache and the optional tick recorder. It exposes one trade iterator to
the pipeline and hides every concurrency detail behind it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..core.config import ExchangeConfig, OrderFlowConfig, PipelineConfig
from ..core.models import BookSnapshot, Candle, DepthUpdate, Trade
from ..core.timeframes import Timeframe
from .feeds import (
    BinanceDepthFeed,
    BinanceTradeFeed,
    OHLCVFeed,
    ReplayFeed,
    SyntheticFeed,
    TradeFeed,
)
from .order_book import OrderBook
from .recorder import TickRecorder
from .timestamp_sync import TimestampSync

__all__ = ["IngestionHub", "build_trade_feed"]

logger = logging.getLogger(__name__)


class IngestionHub:
    """Owns every layer 1 resource for one pipeline run."""

    def __init__(
        self,
        config: OrderFlowConfig,
        trade_feed: TradeFeed | None = None,
        depth_feed: Any | None = None,
        ohlcv_feed: OHLCVFeed | None = None,
        record: bool | None = None,
    ) -> None:
        """Create the hub.

        Args:
            config: Full engine configuration.
            trade_feed: Trade source override. Defaults to the feed implied by
                ``pipeline.mode``.
            depth_feed: Order book source override. Defaults to the live
                websocket depth feed when running live.
            ohlcv_feed: Higher timeframe candle source override. Defaults to the
                REST poller when running live.
            record: Force tick recording on or off. Defaults to
                ``pipeline.record_path`` being set.
        """
        self.config = config
        self.pipeline: PipelineConfig = config.pipeline
        self.exchange: ExchangeConfig = config.exchange
        self.timestamps = TimestampSync()
        self.trade_feed: TradeFeed = trade_feed or build_trade_feed(config)
        self.depth_feed = depth_feed if depth_feed is not None else self._build_depth_feed(config)
        self.ohlcv_feed = ohlcv_feed if ohlcv_feed is not None else self._build_ohlcv_feed(config)
        self.book = OrderBook(config.exchange.depth_levels)
        self.context_candles: dict[Timeframe, list[Candle]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._recorder: TickRecorder | None = None
        should_record = self.pipeline.record_path if record is None else record
        if should_record and self.pipeline.record_path:
            self._recorder = TickRecorder(self.pipeline.record_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Seed higher timeframe context and start background feed tasks."""
        await self._load_context_candles()
        if self.depth_feed is not None:
            self._tasks.append(asyncio.create_task(self._consume_depth(), name="orderflow-depth"))

    async def aclose(self) -> None:
        """Cancel background tasks, flush the recorder and close feeds."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
        if self._recorder is not None:
            self._recorder.close()
        await self.trade_feed.aclose()
        if self.depth_feed is not None:
            await self.depth_feed.aclose()

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------
    async def trades(self) -> AsyncIterator[Trade]:
        """Yield normalised trades, refreshing book state and recording ticks."""
        snapshot_source = getattr(self.trade_feed, "snapshot", None)
        last_book_refresh = float("-inf")
        last_recorded_book = float("-inf")

        async for trade in self.trade_feed.stream():
            normalised = Trade(
                symbol=trade.symbol,
                ts=self.timestamps.normalize(self.trade_feed.name, trade.ts),
                price=trade.price,
                size=trade.size,
                side=trade.side,
                source=trade.source,
            )

            if self._recorder is not None:
                self._recorder.record_trade(normalised)
                if normalised.ts - last_recorded_book >= self._recorder.book_interval:
                    last_recorded_book = normalised.ts
                    self._recorder.record_book(self.snapshot(), normalised.ts)

            if snapshot_source is not None and normalised.ts - last_book_refresh >= 1.0:
                # Offline feeds maintain their own book; poll it instead of
                # subscribing to a second stream.
                self._absorb(snapshot_source())
                last_book_refresh = normalised.ts

            yield normalised

    def snapshot(self) -> BookSnapshot:
        """Return the current order book."""
        return self.book.snapshot()

    def diagnostics(self) -> dict[str, Any]:
        """Ingestion health: timestamp report plus optional backpressure info."""
        return {
            "mode": self.pipeline.mode,
            "symbol": self.exchange.symbol,
            "trade_feed": self.trade_feed.name,
            "depth_feed": getattr(self.depth_feed, "name", None),
            "recorded": self._recorder.path if self._recorder else None,
            "timestamps": self.timestamps.report(),
            "book_levels": len(self.book.snapshot().bids),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _absorb(self, snapshot: BookSnapshot | None) -> None:
        """Push a whole book snapshot into the local order book."""
        if not snapshot:
            return
        self.book.apply(
            DepthUpdate(
                ts=self.book.last_update,
                bids=tuple(snapshot.bids),
                asks=tuple(snapshot.asks),
                is_snapshot=True,
            )
        )

    async def _consume_depth(self) -> None:
        """Keep the local book in sync with the depth feed."""
        try:
            async for update in self.depth_feed.stream():
                self.book.apply(update)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - ingestion must degrade, not crash
            logger.warning("depth feed stopped: %s", exc)

    async def _load_context_candles(self) -> None:
        """Fetch higher timeframe candles once, ignoring network failures."""
        if self.ohlcv_feed is None:
            return
        for timeframe in self.config.analysis_timeframes():
            if timeframe is self.config.timeframe:
                continue
            try:
                self.context_candles[timeframe] = await self.ohlcv_feed.history(
                    limit=max(self.pipeline.history_bars, 20)
                )
            except Exception as exc:  # noqa: BLE001 - context is best effort
                logger.warning("could not load %s context: %s", timeframe, exc)
                self.context_candles[timeframe] = []

    @staticmethod
    def _build_depth_feed(config: OrderFlowConfig) -> Any | None:
        """Live runs get a websocket depth feed; other modes rely on their own."""
        if config.pipeline.mode == "live":
            return BinanceDepthFeed(config.exchange)
        return None

    @staticmethod
    def _build_ohlcv_feed(config: OrderFlowConfig) -> OHLCVFeed | None:
        """Only live runs fetch higher timeframe history over REST."""
        if config.pipeline.mode == "live":
            return OHLCVFeed(config.exchange, config.timeframe, limit=config.pipeline.history_bars * 3)
        return None


def build_trade_feed(config: OrderFlowConfig) -> TradeFeed:
    """Create the trade feed implied by ``config.pipeline.mode``.

    Args:
        config: Full engine configuration.

    Returns:
        A live websocket feed, a replay feed or the synthetic generator.

    Raises:
        ValueError: If the mode is unknown or replay has no source path.
    """
    mode = config.pipeline.mode.lower()
    if mode == "live":
        return BinanceTradeFeed(config.exchange)
    if mode == "replay":
        if not config.pipeline.replay_path:
            raise ValueError("pipeline.replay_path is required when mode = replay")
        start, end = config.pipeline.replay_window
        return ReplayFeed(
            config.pipeline.replay_path,
            symbol=config.exchange.symbol,
            price_decimals=config.exchange.price_precision,
            depth_levels=config.exchange.depth_levels,
            start=start,
            end=end,
        )
    if mode == "synthetic":
        return SyntheticFeed(config.exchange, config.timeframe, bars=120)
    raise ValueError(f"Unknown pipeline mode {config.pipeline.mode!r}; use live, replay or synthetic")