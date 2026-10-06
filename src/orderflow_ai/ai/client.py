"""High level AI client.

:class:`AIClient` ties the prompt builder, the provider and the validator
together. It never returns an unvalidated signal: when the model reply fails the
contract, the caller gets a :class:`ValidationResult` explaining why.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from ..core.analysis import Analysis
from ..core.config import AIConfig
from ..output.prompt import PromptBuild, PromptBuilder
from ..output.schema import SignalValidator, ValidationResult
from .providers import BaseProvider, ProviderError, ProviderResponse, build_provider

__all__ = ["AIClient", "AIError"]

logger = logging.getLogger(__name__)


class AIError(RuntimeError):
    """Raised when the AI stage cannot produce a validated signal."""


class AIClient:
    """Builds prompts, calls a provider and validates the reply."""

    def __init__(
        self,
        config: AIConfig | None = None,
        provider: BaseProvider | str | None = None,
        prompt_builder: PromptBuilder | None = None,
        validator: SignalValidator | None = None,
        json_mode: bool = True,
    ) -> None:
        """Create the client.

        Args:
            config: Provider settings. Defaults are used when omitted.
            provider: Provider override: an instance (mainly for tests) or a
                provider name such as ``"openai"``, which is what the
                documented Python API passes.
            prompt_builder: Prompt source override.
            validator: Signal validator override.
            json_mode: Ask the provider for a JSON reply when supported.
        """
        self._config = config or AIConfig()
        if isinstance(provider, str):
            self._config = dataclasses.replace(self._config, provider=provider)
            self._provider = build_provider(self._config)
        else:
            self._provider = provider or build_provider(self._config)
        self._prompts = prompt_builder or PromptBuilder()
        self._validator = validator or SignalValidator(
            min_reward_risk=self._config.min_reward_risk
        )
        self._json_mode = json_mode

    @property
    def provider(self) -> BaseProvider:
        """Provider in use."""
        return self._provider

    @property
    def prompts(self) -> PromptBuilder:
        """Prompt builder in use."""
        return self._prompts

    @property
    def validator(self) -> SignalValidator:
        """Validator in use."""
        return self._validator

    async def complete(
        self, messages: list[dict[str, str]], client: Any | None = None
    ) -> ProviderResponse:
        """Send raw messages to the provider."""
        return await self._provider.complete(messages, self._json_mode, client=client)

    async def analyze(
        self,
        payload: dict[str, Any],
        known_patterns: tuple[str, ...] = (),
        client: Any | None = None,
    ) -> ValidationResult:
        """Ask the model to grade a prepared payload.

        Args:
            payload: Optimised payload as produced by layer 5.
            known_patterns: Pattern names in the payload, used to detect
                hallucinated patterns in the reply.
            client: Optional HTTP client, mainly for tests.

        Returns:
            The validation result. Use ``result.raise_for_errors()`` to turn it
            into a :class:`~orderflow_ai.output.schema.TradingSignal`.
        """
        messages = [
            {"role": "system", "content": self._prompts.static_block()},
            {"role": "user", "content": self._prompts.payload_builder.optimizer.serialize(payload)},
        ]
        response = await self.complete(messages, client=client)
        return self._validator.validate(response.json(), known_patterns=known_patterns)

    async def analyze_analysis(
        self, analysis: Analysis, client: Any | None = None
    ) -> tuple[ValidationResult, PromptBuild]:
        """Build the prompt from ``analysis``, call the model and validate it.

        Returns:
            The validation result and the prompt that produced it, so callers
            can log payload size and cache keys together with the signal.
        """
        prompt = self._prompts.build(analysis)
        try:
            response = await self.complete(
                [message.to_dict() for message in prompt.messages], client=client
            )
        except ProviderError as exc:
            logger.error("AI provider call failed: %s", exc)
            raise AIError(str(exc)) from exc

        result = self._validator.validate(
            response.json(), known_patterns=analysis.pattern_names
        )
        return result, prompt