"""Layer 5: tiers, optimisation, prompt assembly and signal validation."""

from __future__ import annotations

import json

import pytest

from orderflow_ai.core.config import OptimizationConfig
from orderflow_ai.output import (
    PayloadBuilder,
    PayloadOptimizer,
    Prioritizer,
    PromptBuilder,
    SignalSchema,
    SignalValidator,
    Summarizer,
    TierSelector,
    dumps,
    estimate_tokens,
)
from orderflow_ai.output.formatter import encode_arrays
from orderflow_ai.output.optimizer import _delta_encode, _drop_empty, _shorten_keys
from orderflow_ai.output.prompt import LEGEND
from orderflow_ai.output.tiers import TIER_FULL, TIER_MINIMAL, TIER_STANDARD
from tests.conftest import default_reply


class TestTierSelection:
    def test_tier_matches_the_documented_rules(self, config) -> None:
        from orderflow_ai.core.pipeline import OrderFlowPipeline
        from orderflow_ai.ingestion.feeds.synthetic import SyntheticFeed
        from tests.conftest import run_async

        feed = SyntheticFeed(config.exchange, config.timeframe, bars=60, seed=3)
        pipeline = OrderFlowPipeline(config, trade_feed=feed)
        analyses = run_async(pipeline.collect(limit=40))
        selector = TierSelector("auto")

        for item in analyses:
            tier = selector.select(item).tier
            if not item.has_pattern or item.confidence < 50:
                assert tier == TIER_FULL
            elif len(item.patterns.patterns) >= 2 and item.confidence >= 80:
                assert tier == TIER_MINIMAL
            else:
                assert tier == TIER_STANDARD

    def test_reports_a_reason(self, analysis) -> None:
        tier = TierSelector("auto").select(analysis)
        assert tier.reason
        assert tier.target_tokens > 0

    def test_fixed_tier_is_respected(self, analysis) -> None:
        assert TierSelector("minimal").select(analysis).tier == TIER_MINIMAL

    def test_exposes_the_configured_mode(self) -> None:
        assert TierSelector("auto").configured == "auto"


class TestOptimizer:
    def test_shortens_known_keys(self) -> None:
        assert _shorten_keys({"confidence": 1, "unknown_key": 2}) == {"cf": 1, "unknown_key": 2}

    def test_shortens_nested_structures(self) -> None:
        assert _shorten_keys({"levels": [{"price": 1.0}]}) == {"lv": [{"px": 1.0}]}

    def test_drops_none_and_empty_containers(self) -> None:
        cleaned = _drop_empty({"a": None, "b": {}, "c": [], "d": 0, "e": False})
        assert cleaned == {"d": 0, "e": False}

    def test_delta_encodes_numeric_series(self) -> None:
        assert _delta_encode({"ev": [1.0, 3.0, 6.0]}) == {"ev": [1.0, 2.0, 3.0]}

    def test_delta_encoding_ignores_non_numeric_series(self) -> None:
        assert _delta_encode({"ev": ["a", "b"]}) == {"ev": ["a", "b"]}

    def test_compact_json_has_no_whitespace(self) -> None:
        assert dumps({"a": 1, "b": 2}, compact=True) == '{"a":1,"b":2}'
        assert " " in dumps({"a": 1, "b": 2}, compact=False)

    def test_reports_size_reduction(self, analysis) -> None:
        payload = PayloadBuilder().build(analysis)
        optimized, report = PayloadOptimizer().optimize(encode_arrays(payload), baseline=payload)
        assert report.final_bytes < report.raw_bytes
        assert 0.0 < report.reduction < 1.0
        assert report.saved_bytes == report.raw_bytes - report.final_bytes
        assert "compact_json" in report.strategies
        assert isinstance(optimized, dict)

    def test_strategies_can_be_disabled(self, analysis) -> None:
        payload = {"market": {"symbol": "BTCUSDT", "price": 1.23456789}}
        optimizer = PayloadOptimizer(
            compact_json=False,
            short_keys=False,
            drop_nulls=False,
            round_numbers=False,
            delta_encode=False,
        )
        optimized, report = optimizer.optimize(payload)
        assert optimized == payload
        assert report.strategies == ()

    def test_token_estimate_scales_with_size(self) -> None:
        assert estimate_tokens("a" * 400) == 100
        assert estimate_tokens("") == 1

    def test_serialisation_round_trips(self, analysis) -> None:
        payload = PayloadBuilder().build_optimized(analysis).payload
        assert json.loads(json.dumps(payload)) == payload


class TestArrayEncoding:
    def test_shortens_section_names(self) -> None:
        encoded = encode_arrays({"market": {}, "patterns": []})
        assert set(encoded) == {"m", "pat"}

    def test_unknown_sections_pass_through(self) -> None:
        assert encode_arrays({"custom": {"a": 1}}) == {"custom": {"a": 1}}

    def test_encodes_the_market_section(self) -> None:
        section = {
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "bar_open": "2024-01-01T00:00+00:00",
            "price": 1.23456789,
            "tier": "standard",
        }
        assert encode_arrays({"market": section})["m"] == [
            "BTCUSDT",
            "1m",
            "2024-01-01T00:00+00:00",
            1.2346,
            "standard",
        ]

    def test_encodes_patterns_into_rows(self) -> None:
        patterns = [
            {
                "name": "absorption",
                "price": 1.08523456,
                "direction": "bullish",
                "confidence": 78.4,
                "bars_ago": 0,
            }
        ]
        assert encode_arrays({"patterns": patterns})["pat"] == [
            ["absorption", 1.0852, "bullish", 78.0, 0]
        ]

    def test_encodes_levels(self) -> None:
        section = {"support": [99.123456], "resistance": [101.5], "poc": 100.0}
        assert encode_arrays({"levels": section})["lvl"] == [[99.1235], [101.5], 100.0]

    def test_keeps_direction_positions_when_divergence_is_absent(self) -> None:
        section = {
            "delta": 1.0,
            "delta_ratio": 0.1,
            "direction": "bullish",
            "cvd": 2.0,
            "cvd_slope": 0.3,
            "momentum": "neutral",
            "divergence": None,
        }
        encoded = encode_arrays({"direction": section})["d"]
        assert len(encoded) == 7
        assert encoded[-1] == ""

    def test_pressure_encodes_an_empty_top_when_nothing_is_imbalanced(self) -> None:
        section = {
            "imbalance": {"strongest": None},
            "orderbook": {
                "bid_depth": 1.0,
                "ask_depth": 2.0,
                "imbalance_ratio": 0.5,
                "direction": "ask",
            },
        }
        encoded = encode_arrays({"pressure": section})["p"]
        assert encoded["top"] == []
        assert encoded["ob"][3] == "ask"


class TestPayloadBuilder:
    def test_minimal_tier_omits_the_heavy_sections(self, analysis) -> None:
        payload = PayloadBuilder().build(analysis, tier=TIER_MINIMAL)
        assert set(payload) == {"market", "direction", "patterns", "levels", "context"}
        assert "footprint" not in dumps(payload)

    def test_standard_tier_adds_structure(self, analysis) -> None:
        payload = PayloadBuilder().build(analysis, tier=TIER_STANDARD)
        assert "structure" in payload
        assert "fallback" not in payload

    def test_full_tier_omits_the_fallback_when_a_pattern_exists(self, analysis) -> None:
        from orderflow_ai.output.tiers import TIER_FULL as FULL

        payload = PayloadBuilder().build(analysis, tier=FULL)
        if analysis.has_pattern:
            assert "fallback" not in payload

    def test_report_and_tier_are_reported(self, analysis) -> None:
        build = PayloadBuilder().build_optimized(analysis)
        assert build.tier in {TIER_MINIMAL, TIER_STANDARD, TIER_FULL}
        assert build.token_estimate > 0
        assert build.priority.lead

    def test_serialises_without_whitespace(self, analysis) -> None:
        text = PayloadBuilder().serialize(analysis)
        assert ", " not in text
        assert json.loads(text)["m"][0] == analysis.symbol

    def test_respects_the_optimizer_configuration(self, analysis) -> None:
        optimizer = PayloadOptimizer(
            compact_json=OptimizationConfig().compact_json, short_keys=False
        )
        payload = PayloadBuilder(optimizer=optimizer).build_optimized(analysis).payload
        assert "confidence" not in dumps(payload)


class TestSummarizer:
    def test_market_section(self, analysis) -> None:
        market = Summarizer().market(analysis)
        assert market["symbol"] == analysis.symbol
        assert market["timeframe"] == str(analysis.timeframe)

    def test_levels_section_respects_the_limit(self, analysis) -> None:
        levels = Summarizer(max_levels=1).levels(analysis.context)
        assert len(levels["support"]) <= 1
        assert len(levels["resistance"]) <= 1

    def test_patterns_section_caps_the_count(self, analysis) -> None:
        patterns = Summarizer(max_patterns=1).patterns(analysis.patterns.patterns)
        assert len(patterns) <= 1

    def test_fallback_section_is_none_with_a_pattern(self, analysis) -> None:
        if analysis.has_pattern:
            assert Summarizer().fallback(analysis.patterns) is None


class TestPrioritizer:
    def test_ranks_evidence(self, analysis) -> None:
        result = Prioritizer(Summarizer()).prioritise(analysis, TIER_STANDARD)
        assert result.lead in result.included or result.lead == "none"
        assert result.dropped_count >= 0

    def test_summarises_the_leading_evidence(self, analysis) -> None:
        result = Prioritizer(Summarizer()).prioritise(analysis)
        assert isinstance(result.summary, str)
        assert result.summary


class TestPromptBuilder:
    def test_static_block_is_stable(self) -> None:
        builder = PromptBuilder()
        assert builder.static_block() == builder.static_block()
        assert builder.static_hash() == PromptBuilder().static_hash()

    def test_legend_documents_every_section(self) -> None:
        for key in ("m=", "d=", "pat=", "lvl=", "c=", "s=", "p.ob=", "q=", "fb="):
            assert key in LEGEND

    def test_builds_system_and_user_messages(self, analysis) -> None:
        prompt = PromptBuilder().build(analysis)
        assert [message.role for message in prompt.messages] == ["system", "user"]
        assert prompt.token_estimate > 0
        assert json.loads(prompt.messages[1].content)["m"][0] == analysis.symbol

    def test_reports_the_static_hash_for_prompt_caching(self, analysis) -> None:
        first = PromptBuilder().build(analysis)
        second = PromptBuilder().build(analysis)
        assert first.static_hash == second.static_hash


class TestSignalSchema:
    def test_accepts_a_valid_response(self) -> None:
        assert SignalSchema().validate_shape(default_reply()) == []

    def test_requires_the_documented_fields(self) -> None:
        errors = SignalSchema().validate_shape({"bias": "bullish"})
        assert any("missing required field" in error for error in errors)

    def test_rejects_unknown_enum_values(self) -> None:
        reply = default_reply() | {"signal": "HODL"}
        assert any("signal must be one of" in error for error in SignalSchema().validate_shape(reply))

    def test_rejects_out_of_range_confidence(self) -> None:
        reply = default_reply() | {"confidence": 140}
        assert any("between 0 and 100" in error for error in SignalSchema().validate_shape(reply))

    def test_rejects_empty_reasoning(self) -> None:
        reply = default_reply() | {"reasoning": "   "}
        assert any("reasoning" in error for error in SignalSchema().validate_shape(reply))

    def test_rejects_non_object_responses(self) -> None:
        assert SignalSchema().validate_shape([1, 2, 3])

    def test_json_schema_is_self_describing(self) -> None:
        schema = SignalSchema().json_schema
        assert schema["type"] == "object"
        assert "bias" in schema["properties"]


class TestSignalValidator:
    def test_accepts_a_valid_directional_signal(self) -> None:
        result = SignalValidator().validate(default_reply())
        assert result.valid
        assert result.signal is not None
        assert result.signal.signal == "BUY"
        assert result.signal.reward_risk == pytest.approx(2.0)

    def test_rejects_a_schema_violation(self) -> None:
        result = SignalValidator().validate({"signal": "BUY"})
        assert not result.valid
        assert result.errors

    def test_rejects_confidence_below_the_floor(self) -> None:
        reply = default_reply() | {"confidence": 30}
        result = SignalValidator(min_confidence=50.0).validate(reply)
        assert not result.valid
        assert any("requires confidence" in error for error in result.errors)

    def test_rejects_a_bias_that_contradicts_the_signal(self) -> None:
        reply = default_reply() | {"bias": "bearish"}
        result = SignalValidator().validate(reply)
        assert not result.valid
        assert any("requires bias bullish" in error for error in result.errors)

    def test_rejects_a_thin_reward_to_risk(self) -> None:
        reply = default_reply() | {"setup": {"entry": 100.0, "stop": 99.0, "target": 100.5, "rr": 0.5}}
        result = SignalValidator(min_reward_risk=1.5).validate(reply)
        assert not result.valid
        assert any("below minimum" in error for error in result.errors)

    def test_derives_reward_risk_from_prices(self) -> None:
        reply = default_reply() | {"setup": {"entry": 100.0, "stop": 99.0, "target": 103.0}}
        result = SignalValidator(min_reward_risk=1.5).validate(reply)
        assert result.valid

    def test_requires_a_setup_for_directional_signals(self) -> None:
        reply = default_reply() | {"setup": None}
        result = SignalValidator().validate(reply)
        assert not result.valid
        assert any("setup.rr" in error for error in result.errors)

    def test_rejects_hallucinated_patterns(self) -> None:
        reply = default_reply() | {"reasoning": "Iceberg accumulation plus an exhaustion move."}
        result = SignalValidator().validate(reply, known_patterns=("absorption",))
        assert not result.valid
        assert any("unknown patterns" in error for error in result.errors)

    def test_accepts_patterns_present_in_the_payload(self) -> None:
        result = SignalValidator().validate(default_reply(), known_patterns=("absorption",))
        assert result.valid

    def test_rejects_no_trade_with_directional_confidence(self) -> None:
        reply = default_reply() | {"signal": "NO_TRADE", "bias": "neutral", "confidence": 90}
        result = SignalValidator().validate(reply)
        assert not result.valid
        assert any("NO_TRADE contradicts" in error for error in result.errors)

    def test_accepts_a_wait_signal(self) -> None:
        reply = {
            "bias": "neutral",
            "confidence": 40,
            "signal": "WAIT",
            "primary_evidence": "context",
            "reasoning": "Volatility expanding, no clean level.",
            "wait_for": "break above 101.0",
        }
        result = SignalValidator().validate(reply)
        assert result.valid
        assert result.signal.wait_for == "break above 101.0"

    def test_warns_without_rejecting(self) -> None:
        reply = default_reply() | {"invalidation": ""}
        result = SignalValidator().validate(reply)
        assert result.valid
        assert any("invalidation" in warning for warning in result.warnings)

    def test_raise_for_errors_returns_the_signal(self) -> None:
        assert SignalValidator().validate(default_reply()).raise_for_errors().signal == "BUY"

    def test_raise_for_errors_on_a_failure(self) -> None:
        from orderflow_ai.output.schema import SignalError

        with pytest.raises(SignalError):
            SignalValidator().validate({}).raise_for_errors()