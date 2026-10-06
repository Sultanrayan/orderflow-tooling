"""Configuration loading, coercion and validation."""

from __future__ import annotations

import pytest

from orderflow_ai.core.config import (
    AIConfig,
    ExchangeConfig,
    OptimizationConfig,
    OrderFlowConfig,
    PatternsConfig,
    PipelineConfig,
    PrimitivesConfig,
    load_config,
)
from orderflow_ai.core.timeframes import Timeframe

MINIMAL_YAML = """
exchange:
  symbol: ETHUSDT
  price_precision: 3
pipeline:
  timeframe: 5m
  history_bars: 40
patterns:
  min_confidence: 55
"""


def write(tmp_path, text: str):
    """Write a YAML file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestDefaults:
    def test_defaults_are_complete(self) -> None:
        config = OrderFlowConfig.from_mapping(None)
        assert isinstance(config.exchange, ExchangeConfig)
        assert isinstance(config.pipeline, PipelineConfig)
        assert isinstance(config.primitives, PrimitivesConfig)
        assert isinstance(config.patterns, PatternsConfig)
        assert isinstance(config.ai, AIConfig)
        assert isinstance(config.optimization, OptimizationConfig)

    def test_shortcuts_expose_symbol_and_timeframe(self) -> None:
        config = OrderFlowConfig.from_mapping(None)
        assert config.symbol == config.exchange.symbol
        assert config.timeframe is config.pipeline.timeframe

    def test_defaults_parse_timeframes(self) -> None:
        assert OrderFlowConfig.from_mapping(None).timeframe is Timeframe.M1


class TestLoading:
    def test_loads_a_yaml_file(self, tmp_path) -> None:
        config = OrderFlowConfig.from_yaml(write(tmp_path, MINIMAL_YAML))
        assert config.exchange.symbol == "ETHUSDT"
        assert config.exchange.price_precision == 3
        assert config.timeframe is Timeframe.M5
        assert config.pipeline.history_bars == 40
        assert config.patterns.min_confidence == 55

    def test_unspecified_sections_keep_their_defaults(self, tmp_path) -> None:
        config = OrderFlowConfig.from_yaml(write(tmp_path, MINIMAL_YAML))
        assert config.exchange.name == ExchangeConfig().name
        assert config.ai.model == AIConfig().model

    def test_load_accepts_a_path_string(self, tmp_path) -> None:
        config = load_config(str(write(tmp_path, MINIMAL_YAML)))
        assert config.symbol == "ETHUSDT"

    def test_load_passes_through_instances(self) -> None:
        config = OrderFlowConfig.from_mapping(None)
        assert OrderFlowConfig.load(config) is config

    def test_load_accepts_a_mapping(self) -> None:
        assert OrderFlowConfig.load({"exchange": {"symbol": "SOLUSDT"}}).symbol == "SOLUSDT"

    def test_missing_file_is_reported(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            OrderFlowConfig.from_yaml(tmp_path / "absent.yaml")

    def test_empty_file_yields_defaults(self, tmp_path) -> None:
        assert OrderFlowConfig.from_yaml(write(tmp_path, "")).timeframe is Timeframe.M1

    def test_non_mapping_root_is_rejected(self, tmp_path) -> None:
        with pytest.raises(TypeError):
            OrderFlowConfig.from_yaml(write(tmp_path, "- one\n- two\n"))


class TestValidation:
    def test_unknown_section_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown configuration sections"):
            OrderFlowConfig.from_mapping({"nonsense": {}})

    def test_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown keys"):
            OrderFlowConfig.from_mapping({"exchange": {"ticksize": 1}})

    def test_non_mapping_section_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            OrderFlowConfig.from_mapping({"exchange": "binance"})

    def test_unknown_timeframe_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            OrderFlowConfig.from_mapping({"pipeline": {"timeframe": "7m"}})

    def test_null_values_keep_defaults(self) -> None:
        config = OrderFlowConfig.from_mapping({"exchange": {"symbol": None}})
        assert config.symbol == ExchangeConfig().symbol


class TestCoercion:
    def test_strings_become_numbers(self, tmp_path) -> None:
        config = OrderFlowConfig.from_yaml(write(tmp_path, "pipeline:\n  history_bars: '12'\n"))
        assert config.pipeline.history_bars == 12

    def test_text_becomes_a_boolean(self, tmp_path) -> None:
        config = OrderFlowConfig.from_yaml(
            write(tmp_path, "optimization:\n  short_keys: 'false'\n")
        )
        assert config.optimization.short_keys is False

    def test_booleans_stay_booleans(self) -> None:
        assert OrderFlowConfig.from_mapping({"optimization": {"delta_encode": False}}).optimization.delta_encode is False

    def test_context_timeframes_become_a_tuple(self) -> None:
        config = OrderFlowConfig.from_mapping({"pipeline": {"context_timeframes": ["5m", "1h"]}})
        assert config.pipeline.context_timeframes == ("5m", "1h")


class TestPipelineConfig:
    def test_replay_window_defaults_to_open(self) -> None:
        assert PipelineConfig().replay_window == (None, None)

    def test_replay_window_parses_iso_timestamps(self) -> None:
        config = PipelineConfig(start="2024-01-01T00:00:00+00:00", end="2024-01-02T00:00:00+00:00")
        start, end = config.replay_window
        assert start == pytest.approx(1_704_067_200.0)
        assert end == pytest.approx(1_704_153_600.0)
        assert end > start

    def test_is_live_flag(self) -> None:
        assert PipelineConfig(mode="live").is_live
        assert not PipelineConfig(mode="synthetic").is_live


class TestAnalysisTimeframes:
    def test_base_timeframe_comes_first(self) -> None:
        config = OrderFlowConfig.from_mapping({"pipeline": {"timeframe": "1m"}})
        assert config.analysis_timeframes()[0] is Timeframe.M1

    def test_shorter_timeframes_are_dropped(self) -> None:
        config = OrderFlowConfig.from_mapping(
            {"pipeline": {"timeframe": "5m", "context_timeframes": ["1m", "1h"]}}
        )
        assert config.analysis_timeframes() == (Timeframe.M5, Timeframe.H1)

    def test_duplicates_are_removed(self) -> None:
        config = OrderFlowConfig.from_mapping(
            {"pipeline": {"timeframe": "1m", "context_timeframes": ["1m", "1m"]}}
        )
        assert config.analysis_timeframes() == (Timeframe.M1,)


class TestAIConfig:
    def test_api_key_falls_back_to_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert AIConfig(provider="openai").resolved_api_key == "sk-test"

    def test_explicit_api_key_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert AIConfig(provider="openai", api_key="sk-explicit").resolved_api_key == "sk-explicit"

    def test_base_url_falls_back_to_the_provider_default(self, monkeypatch) -> None:
        for name in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "OLLAMA_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        assert AIConfig(provider="openai").resolved_base_url == "https://api.openai.com/v1"
        assert AIConfig(provider="anthropic").resolved_base_url.startswith("https://api.anthropic.com")
        assert AIConfig(provider="ollama").resolved_base_url.startswith("http://localhost")

    def test_base_url_falls_back_to_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
        assert AIConfig(provider="ollama").resolved_base_url == "http://ollama:11434"

    def test_only_ollama_runs_without_a_key(self) -> None:
        assert AIConfig(provider="openai").requires_api_key
        assert not AIConfig(provider="ollama").requires_api_key