"""AI stage: provider request shapes and client side validation."""

from __future__ import annotations

import json

import pytest

from orderflow_ai.ai import AIClient, AIError, build_provider
from orderflow_ai.ai.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
)
from orderflow_ai.core.config import AIConfig
from orderflow_ai.output.schema import SignalValidator
from tests.conftest import StubProvider, default_reply, reply_for


class TestProviderFactory:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("openai", OpenAIProvider),
            ("anthropic", AnthropicProvider),
            ("ollama", OllamaProvider),
        ],
    )
    def test_builds_each_supported_provider(self, name: str, expected: type) -> None:
        provider = build_provider(AIConfig(provider=name))
        assert isinstance(provider, expected)
        assert provider.name == name

    def test_is_case_insensitive(self) -> None:
        assert isinstance(build_provider(AIConfig(provider="OpenAI")), OpenAIProvider)

    def test_rejects_unknown_providers(self) -> None:
        with pytest.raises(ValueError, match="Unknown AI provider"):
            build_provider(AIConfig(provider="gpt"))


class TestOpenAIProvider:
    provider = OpenAIProvider(AIConfig(provider="openai", api_key="sk-test", model="gpt-4o"))
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "payload"}]

    def test_endpoint_and_headers(self) -> None:
        assert self.provider.endpoint().endswith("/chat/completions")
        assert self.provider.headers()["Authorization"] == "Bearer sk-test"

    def test_request_body(self) -> None:
        body = self.provider.build_request(self.messages, json_mode=True)
        assert body["model"] == "gpt-4o"
        assert body["temperature"] == pytest.approx(0.2)
        assert body["max_tokens"] == 300
        assert body["response_format"] == {"type": "json_object"}
        assert body["messages"] == self.messages

    def test_omits_response_format_without_json_mode(self) -> None:
        assert "response_format" not in self.provider.build_request(self.messages, json_mode=False)

    def test_parses_a_reply(self) -> None:
        response = self.provider.parse_response(
            {
                "model": "gpt-4o",
                "choices": [{"message": {"content": '{"bias":"neutral"}'}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 40},
            }
        )
        assert response.json() == {"bias": "neutral"}
        assert response.usage["input_tokens"] == 120

    def test_rejects_an_empty_choice_list(self) -> None:
        with pytest.raises(ProviderError, match="no choices"):
            self.provider.parse_response({"choices": []})


class TestAnthropicProvider:
    provider = AnthropicProvider(AIConfig(provider="anthropic", api_key="sk-ant", model="claude"))
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "payload"}]

    def test_endpoint_and_headers(self) -> None:
        assert self.provider.endpoint().endswith("/messages")
        assert self.provider.headers()["x-api-key"] == "sk-ant"
        assert self.provider.headers()["anthropic-version"] == AnthropicProvider.API_VERSION

    def test_request_splits_the_system_prompt(self) -> None:
        body = self.provider.build_request(self.messages, json_mode=True)
        assert body["system"] == "rules"
        assert body["messages"] == [{"role": "user", "content": "payload"}]
        assert "response_format" not in body

    def test_parses_text_blocks(self) -> None:
        response = self.provider.parse_response(
            {
                "model": "claude",
                "content": [{"type": "text", "text": '{"bias":"bullish"}'}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        assert response.json() == {"bias": "bullish"}
        assert response.usage["output_tokens"] == 5


class TestOllamaProvider:
    provider = OllamaProvider(AIConfig(provider="ollama", model="llama3.1"))

    def test_request_uses_the_local_format_flag(self) -> None:
        body = self.provider.build_request([{"role": "user", "content": "hi"}], json_mode=True)
        assert body["model"] == "llama3.1"
        assert body["format"] == "json"
        assert body["stream"] is False
        assert body["options"]["temperature"] == pytest.approx(0.2)

    def test_needs_no_api_key(self) -> None:
        assert not self.provider.requires_api_key()

    def test_parses_a_reply(self) -> None:
        response = self.provider.parse_response(
            {"message": {"content": '{"signal":"NO_TRADE"}'}, "prompt_eval_count": 7, "eval_count": 3}
        )
        assert response.json() == {"signal": "NO_TRADE"}
        assert response.usage["input_tokens"] == 7


class TestProviderResponse:
    def test_parses_plain_json(self) -> None:
        from orderflow_ai.ai.providers import ProviderResponse

        assert ProviderResponse(content='{"a": 1}', model="m").json() == {"a": 1}

    def test_parses_a_fenced_reply(self) -> None:
        from orderflow_ai.ai.providers import ProviderResponse

        reply = '```json\n{"a": 1}\n```'
        assert ProviderResponse(content=reply, model="m").json() == {"a": 1}

    def test_rejects_invalid_json(self) -> None:
        from orderflow_ai.ai.providers import ProviderResponse

        with pytest.raises(ProviderError, match="not valid JSON"):
            ProviderResponse(content="not json", model="m").json()

    def test_rejects_non_object_json(self) -> None:
        from orderflow_ai.ai.providers import ProviderResponse

        with pytest.raises(ProviderError, match="must be a JSON object"):
            ProviderResponse(content="[1, 2]", model="m").json()


class TestAIClient:
    async def test_validates_a_canned_reply(self, analysis) -> None:
        client = AIClient(provider=StubProvider(reply=reply_for(analysis)))
        result, prompt = await client.analyze_analysis(analysis)

        assert result.valid
        assert result.signal.signal == "BUY"
        assert prompt.tier

    async def test_rejects_an_invalid_reply(self, analysis) -> None:
        client = AIClient(provider=StubProvider(reply={"signal": "BUY"}))
        result, _ = await client.analyze_analysis(analysis)
        assert not result.valid
        assert result.errors

    async def test_detects_hallucinated_patterns(self, analysis) -> None:
        invented = next(
            name for name in ("iceberg", "exhaustion", "trapped_traders") if name not in analysis.pattern_names
        )
        client = AIClient(
            provider=StubProvider(reply=reply_for(analysis, reasoning=f"Clear {invented} setup."))
        )
        result, _ = await client.analyze_analysis(analysis)

        assert not result.valid
        assert any("unknown patterns" in error for error in result.errors)

    async def test_uses_the_configured_reward_risk(self, analysis) -> None:
        reply = default_reply() | {"setup": {"entry": 100.0, "stop": 99.0, "target": 100.4, "rr": 0.4}}
        client = AIClient(
            config=AIConfig(min_reward_risk=2.0),
            provider=StubProvider(reply=reply),
        )
        result, _ = await client.analyze_analysis(analysis)
        assert not result.valid

    async def test_sends_the_optimised_payload(self, analysis, stub_provider) -> None:
        client = AIClient(provider=stub_provider)
        payload = client.prompts.payload_builder.build_optimized(analysis).payload
        await client.analyze(payload, known_patterns=analysis.pattern_names)

        assert stub_provider.requests
        sent = stub_provider.requests[0]["messages"][1]["content"]
        assert json.loads(sent)["m"][0] == analysis.symbol

    async def test_wraps_provider_failures(self, analysis) -> None:
        class Failing(StubProvider):
            async def complete(self, messages, json_mode=True, client=None):
                raise ProviderError("network down")

        client = AIClient(provider=Failing())
        with pytest.raises(AIError, match="network down"):
            await client.analyze_analysis(analysis)

    async def test_requires_a_key_for_keyed_providers(self, monkeypatch, analysis) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client = AIClient(config=AIConfig(provider="openai", api_key=None))
        with pytest.raises(ProviderError, match="requires an API key"):
            await client.complete([{"role": "user", "content": "hi"}])

    def test_accepts_a_provider_name_as_in_the_documented_api(self) -> None:
        client = AIClient(provider="ollama")

        assert isinstance(client.provider, OllamaProvider)
        assert client.prompts

    def test_validator_uses_the_configured_floor(self) -> None:
        client = AIClient(
            config=AIConfig(min_reward_risk=2.5),
            provider=StubProvider(),
            validator=SignalValidator(min_reward_risk=2.5),
        )
        assert client.validator.min_reward_risk == pytest.approx(2.5)