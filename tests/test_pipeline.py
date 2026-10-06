"""End to end pipeline behaviour across the five layers."""

from __future__ import annotations

import json

import pytest

from orderflow_ai.core.pipeline import OrderFlowPipeline
from orderflow_ai.ingestion.feeds.synthetic import SyntheticFeed


def pipeline_for(config, **feed_kwargs) -> OrderFlowPipeline:
    """Build a pipeline driven by the synthetic feed."""
    feed = SyntheticFeed(config.exchange, config.timeframe, **feed_kwargs)
    return OrderFlowPipeline(config, trade_feed=feed)


class TestStreaming:
    async def test_emits_one_analysis_per_closed_bar(self, config) -> None:
        analyses = await pipeline_for(config, bars=20).collect(limit=5)
        assert len(analyses) == 5
        assert [item.open_time for item in analyses] == sorted(item.open_time for item in analyses)

    async def test_honours_the_bar_limit(self, config) -> None:
        analyses = await pipeline_for(config, bars=20).collect(limit=3)
        assert len(analyses) == 3

    async def test_run_once_returns_a_single_analysis(self, config) -> None:
        analysis = await pipeline_for(config, bars=10).run_once()
        assert analysis.symbol == config.symbol

    async def test_run_once_reports_an_empty_feed(self, config, tmp_path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        pipeline = OrderFlowPipeline(config, trade_feed=_replay_feed(empty, config))
        with pytest.raises(RuntimeError, match="no bar closed"):
            await pipeline.run_once()

    async def test_history_stays_within_the_lookback(self, config) -> None:
        pipeline = pipeline_for(config, bars=20)
        await pipeline.collect(limit=8)
        assert len(pipeline.history) <= config.pipeline.history_bars


class TestAnalysisContent:
    async def test_every_layer_is_populated(self, analysis) -> None:
        assert analysis.primitives.structure.footprint.total_volume > 0
        assert analysis.primitives.structure.volume_profile.poc > 0
        assert analysis.patterns.fallback is not None or analysis.patterns.patterns
        assert analysis.context.structure.label
        assert analysis.context.session.primary

    async def test_direction_and_confidence_are_derived(self, analysis) -> None:
        assert analysis.direction in {"bullish", "bearish", "neutral"}
        assert 0 <= analysis.confidence <= 100

    async def test_symbol_and_timeframe_come_from_the_config(self, config) -> None:
        analysis = await pipeline_for(config, bars=5).run_once()
        assert analysis.symbol == "BTCUSDT"
        assert analysis.timeframe == config.timeframe

    async def test_level_proximity_is_bounded(self, analysis) -> None:
        assert 0.0 <= analysis.context.level_proximity <= 1.0


class TestLayerFive:
    async def test_payload_matches_the_prompt_legend(self, config) -> None:
        pipeline = pipeline_for(config, bars=30)
        analysis = await pipeline.run_once()
        prompt = pipeline.build_prompt(analysis)
        payload = json.loads(prompt.messages[1].content)

        assert payload["m"][0] == analysis.symbol
        assert payload["m"][4] in {"minimal", "standard", "full"}
        assert len(payload["d"]) == 7

    async def test_payload_is_reproducible(self, config) -> None:
        pipeline = pipeline_for(config, bars=30)
        analysis = await pipeline.run_once()
        assert pipeline.build_payload(analysis) == pipeline.build_payload(analysis)

    async def test_replay_reproduces_the_same_analysis(self, config, recorded_file) -> None:
        first = await _replay(recorded_file, config, limit=3)
        second = await _replay(recorded_file, config, limit=3)

        assert [item.price for item in first] == [item.price for item in second]
        assert [item.confidence for item in first] == [item.confidence for item in second]

    async def test_diagnostics_report_the_ingestion_state(self, config) -> None:
        pipeline = pipeline_for(config, bars=10)
        await pipeline.collect(limit=2)
        diagnostics = pipeline.diagnostics()

        assert diagnostics["ingestion"]["trade_feed"] == "synthetic"
        assert diagnostics["bars_buffered"] > 0
        assert len(diagnostics["detectors"]) == 7


class TestConfigSources:
    def test_loads_the_shipped_default_config(self) -> None:
        pipeline = OrderFlowPipeline("configs/default.yaml")
        assert pipeline.config.symbol == "BTCUSDT"
        assert pipeline.config.pipeline.mode == "synthetic"

    def test_loads_the_shipped_replay_config(self) -> None:
        pipeline = OrderFlowPipeline("configs/replay.yaml")
        assert pipeline.config.pipeline.mode == "replay"

    def test_accepts_a_config_instance(self, config) -> None:
        assert OrderFlowPipeline(config).config is config

    async def test_replays_a_recorded_file(self, config, recorded_file) -> None:
        analyses = await _replay(recorded_file, config)
        assert analyses
        assert all(item.symbol == "BTCUSDT" for item in analyses)


def _replay_feed(path, config):
    """A replay feed configured like the pipeline that consumes it."""
    from orderflow_ai.ingestion.feeds.replay import ReplayFeed

    return ReplayFeed(
        path,
        symbol=config.symbol,
        price_decimals=config.exchange.price_precision,
        depth_levels=config.exchange.depth_levels,
    )


async def _replay(path, config, limit: int = 3) -> list:
    """Replay a recorded file through a pipeline."""
    return await OrderFlowPipeline(config, trade_feed=_replay_feed(path, config)).collect(limit=limit)


def test_scripts_share_the_documented_entry_points() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("live_run.py", "backtest.py", "collect_data.py", "bootstrap.py"):
        assert (root / "scripts" / name).exists()