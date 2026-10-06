"""The pipeline orchestrator.

:class:`OrderFlowPipeline` connects the five layers: it consumes trades, closes
bars, computes primitives, detects patterns, builds context and hands the result
to layer 5. It is the object the documented Python API iterates over.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from ..context.builder import ContextBuilder
from ..context.models import MarketContext
from ..ingestion.candle_builder import CandleBuilder
from ..ingestion.hub import IngestionHub
from ..ingestion.resampler import resample_bars
from ..output.formatter import PayloadBuild, PayloadBuilder
from ..output.prompt import PromptBuild, PromptBuilder
from ..patterns.engine import PatternEngine
from ..patterns.fallback import FallbackContext
from ..patterns.models import PatternContext
from ..primitives.engine import PrimitiveEngine
from .analysis import Analysis
from .config import OrderFlowConfig
from .models import BookSnapshot, FootprintBar
from .timeframes import Timeframe

__all__ = ["OrderFlowPipeline"]

logger = logging.getLogger(__name__)


class OrderFlowPipeline:
    """Runs the full analysis pipeline over a market data feed."""

    def __init__(
        self,
        config: str | Path | OrderFlowConfig | None = None,
        trade_feed: Any | None = None,
        depth_feed: Any | None = None,
        ohlcv_feed: Any | None = None,
    ) -> None:
        """Create the pipeline.

        Args:
            config: Path to a YAML file, a mapping or an
                :class:`~orderflow_ai.core.config.OrderFlowConfig`.
            trade_feed: Trade source override.
            depth_feed: Order book source override.
            ohlcv_feed: Higher timeframe candle source override.
        """
        self.config = OrderFlowConfig.load(config)
        self.hub = IngestionHub(self.config, trade_feed, depth_feed, ohlcv_feed)

        exchange = self.config.exchange
        self.candles = CandleBuilder(self.config.timeframe, exchange.price_precision)
        self.primitives = PrimitiveEngine(
            self.config.primitives,
            tick_size=exchange.tick_size,
            depth_levels=exchange.depth_levels,
            price_decimals=exchange.price_precision,
        )
        self.patterns = PatternEngine(self.config.patterns)
        self.context = ContextBuilder(
            max_levels=self.config.optimization.max_levels,
            volatility_lookback=self.config.pipeline.history_bars,
        )
        self.payloads = PayloadBuilder()
        self.prompts = PromptBuilder(self.payloads)
        self.history: list[FootprintBar] = []

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    async def stream(self, limit: int | None = None) -> AsyncIterator[Analysis]:
        """Yield one :class:`Analysis` per closed bar until the feed ends.

        Args:
            limit: Stop after this many bars. ``None`` consumes the whole feed.

        Yields:
            One analysis per closed bar, oldest first.
        """
        await self.hub.start()
        produced = 0
        try:
            async for trade in self.hub.trades():
                closed = self.candles.push(trade)
                if closed is not None:
                    produced += 1
                    yield self.analyse_bar(closed, self.hub.snapshot())
                    if limit is not None and produced >= limit:
                        return
            final = self.candles.flush()
            if final is not None:
                yield self.analyse_bar(final, self.hub.snapshot())
        finally:
            await self.hub.aclose()

    async def collect(self, limit: int | None = None) -> list[Analysis]:
        """Consume the feed and return up to ``limit`` analyses.

        Args:
            limit: Stop after this many bars. ``None`` consumes the whole feed.

        Returns:
            The analyses, oldest first.
        """
        return [analysis async for analysis in self.stream(limit=limit)]

    async def run_once(self) -> Analysis:
        """Produce a single analysis, for one-shot checks.

        Raises:
            RuntimeError: If the feed ends before any bar closes.
        """
        analyses = await self.collect(limit=1)
        if not analyses:
            raise RuntimeError("no bar closed during this run")
        return analyses[0]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def analyse_bar(self, bar: FootprintBar, book: BookSnapshot | None = None) -> Analysis:
        """Run layers 2 to 4 for one closed bar and buffer it in the history.

        Args:
            bar: The bar that just closed.
            book: Order book state at the close of ``bar``.

        Returns:
            The complete analysis for the bar.
        """
        history = tuple(self.history)
        snapshot = book or BookSnapshot.empty()
        primitives = self.primitives.compute(bar, history, snapshot)
        bars = (*history, bar)
        context = self.context.build(
            primitives,
            bars,
            self.hub.context_candles or self._resampled_context(bars),
        )
        patterns = self.patterns.detect(
            PatternContext(
                primitives=primitives,
                bars=bars,
                supports=context.levels.support,
                resistances=context.levels.resistance,
                tick_size=self.config.exchange.tick_size or 0.01,
                price_decimals=self.config.exchange.price_precision,
            ),
            _fallback_context(context),
        )

        self._append_history(bar)
        return Analysis(
            symbol=primitives.symbol or self.config.symbol,
            timeframe=bar.timeframe,
            open_time=bar.open_time,
            price=primitives.price,
            primitives=primitives,
            patterns=patterns,
            context=context,
            bars=tuple(self.history),
        )

    def _append_history(self, bar: FootprintBar) -> None:
        """Append ``bar`` to the rolling history, trimming to the lookback."""
        self.history.append(bar)
        overflow = len(self.history) - self.config.pipeline.history_bars
        if overflow > 0:
            del self.history[:overflow]

    def _resampled_context(self, bars: Sequence[FootprintBar]) -> dict[Timeframe, list]:
        """Derive higher timeframe context from base bars when no feed provides it."""
        resampled: dict[Timeframe, list] = {}
        for timeframe in self.config.analysis_timeframes():
            if timeframe is self.config.timeframe:
                continue
            candles = resample_bars(bars, timeframe)
            if candles:
                resampled[timeframe] = candles[-self.config.pipeline.history_bars :]
        return resampled

    # ------------------------------------------------------------------
    # Layer 5
    # ------------------------------------------------------------------
    def build(self, analysis: Analysis, tier: str | None = None) -> PayloadBuild:
        """Build the optimised payload with its report and prioritisation."""
        return self.payloads.build_optimized(analysis, tier)

    def build_payload(self, analysis: Analysis, tier: str | None = None) -> dict[str, Any]:
        """Build the optimised payload for ``analysis``."""
        return self.build(analysis, tier).payload

    def build_prompt(self, analysis: Analysis, tier: str | None = None) -> PromptBuild:
        """Build the chat messages for ``analysis``."""
        return self.prompts.build(analysis, tier)

    def diagnostics(self) -> dict[str, Any]:
        """Ingestion health plus the current history size."""
        return {
            "ingestion": self.hub.diagnostics(),
            "bars_buffered": len(self.history),
            "detectors": self.patterns.detector_names,
        }


def _fallback_context(context: MarketContext) -> FallbackContext:
    """Map layer 4 readings onto the fallback scorer's inputs."""
    return FallbackContext(
        mtf_alignment=context.multi_timeframe.score / 100,
        level_proximity=context.level_proximity,
        volatility_regime=context.volatility.regime,
    )