"""Model providers.

Each provider is a thin HTTP client: the request shapes are small, documented
and stable, so an SDK would add a dependency without adding value. ``httpx`` is
imported lazily so that tests and offline runs never need it.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.config import AIConfig

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderResponse",
    "build_provider",
]

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a provider call fails or returns unusable content."""


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Raw provider reply."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)

    def json(self) -> dict[str, Any]:
        """Parse the content as JSON, tolerating markdown fences."""
        text = self.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            _, _, text = text.partition("\n")
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ProviderError(f"model reply is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProviderError("model reply must be a JSON object")
        return payload


class BaseProvider(ABC):
    """Interface implemented by every provider."""

    name = "provider"

    def __init__(self, config: AIConfig | None = None) -> None:
        """Create the provider.

        Args:
            config: Provider settings. Defaults are used when omitted, which is
                also what test doubles rely on.
        """
        self._config = config or AIConfig()

    @property
    def model(self) -> str:
        """Model identifier used for requests."""
        return self._config.model

    @property
    def base_url(self) -> str:
        """Endpoint base URL."""
        return self._config.resolved_base_url.rstrip("/")

    @abstractmethod
    def build_request(self, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        """Return the JSON body for a chat completion request."""

    @abstractmethod
    def endpoint(self) -> str:
        """Full URL of the chat completion endpoint."""

    @abstractmethod
    def headers(self) -> dict[str, str]:
        """HTTP headers required by the provider."""

    @abstractmethod
    def parse_response(self, payload: dict[str, Any]) -> ProviderResponse:
        """Turn the provider's JSON reply into a :class:`ProviderResponse`."""

    def request_kwargs(self) -> dict[str, Any]:
        """Sampling parameters shared by all providers."""
        return {
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
        }

    def requires_api_key(self) -> bool:
        """``True`` when a missing API key must fail fast."""
        return self._config.requires_api_key

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        client: Any | None = None,
    ) -> ProviderResponse:
        """Send a chat completion request and return the reply.

        Args:
            messages: Chat messages as ``{"role": ..., "content": ...}`` dicts.
            json_mode: Ask the provider for a JSON object reply when supported.
            client: Optional pre-built ``httpx.AsyncClient``, mainly for tests.

        Returns:
            The parsed provider response.

        Raises:
            ProviderError: On transport errors, HTTP errors or a missing key.
        """
        if self.requires_api_key() and not self._config.resolved_api_key:
            raise ProviderError(f"{self.name} provider requires an API key")

        import httpx

        body = self.build_request(messages, json_mode)
        url, headers = self.endpoint(), self.headers()
        timeout = self._config.timeout_seconds

        owned_client = client is None
        http = client or httpx.AsyncClient(timeout=timeout)
        try:
            response = await http.post(url, json=body, headers=headers)
            response.raise_for_status()
            return self.parse_response(response.json())
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"{self.name} returned {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc
        finally:
            if owned_client:
                await http.aclose()


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions, including any OpenAI compatible endpoint."""

    name = "openai"

    def endpoint(self) -> str:
        """Chat completions endpoint."""
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        """Bearer authentication plus the JSON content type."""
        return {
            "Authorization": f"Bearer {self._config.resolved_api_key}",
            "Content-Type": "application/json",
        }

    def build_request(self, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        """Request body, with ``response_format`` when JSON mode is on."""
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            **self.request_kwargs(),
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def parse_response(self, payload: dict[str, Any]) -> ProviderResponse:
        """Extract the first choice and its token usage."""
        choices = payload.get("choices") or []
        if not choices:
            raise ProviderError("openai response contained no choices")
        message = choices[0].get("message") or {}
        usage = payload.get("usage") or {}
        return ProviderResponse(
            content=message.get("content") or "",
            model=payload.get("model") or self._config.model,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
        )


class AnthropicProvider(BaseProvider):
    """Anthropic messages API."""

    name = "anthropic"
    API_VERSION = "2023-06-01"
    MAX_TOKENS = 1024

    def endpoint(self) -> str:
        """Messages endpoint."""
        return f"{self.base_url}/messages"

    def headers(self) -> dict[str, str]:
        """API key, version header and JSON content type."""
        return {
            "x-api-key": self._config.resolved_api_key or "",
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }

    def build_request(self, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        """Request body with the system prompt split out of the messages."""
        system = "\n".join(
            message["content"] for message in messages if message["role"] == "system"
        )
        conversation = [message for message in messages if message["role"] != "system"]
        return {
            "model": self._config.model,
            "system": system,
            "messages": conversation,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens or self.MAX_TOKENS,
        }

    def parse_response(self, payload: dict[str, Any]) -> ProviderResponse:
        """Join the text blocks of the reply."""
        blocks = payload.get("content") or []
        content = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        usage = payload.get("usage") or {}
        return ProviderResponse(
            content=content,
            model=payload.get("model") or self._config.model,
            usage={
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            },
        )


class OllamaProvider(BaseProvider):
    """Local Ollama server, no credentials required."""

    name = "ollama"

    def endpoint(self) -> str:
        """Chat endpoint."""
        return f"{self.base_url}/api/chat"

    def headers(self) -> dict[str, str]:
        """Ollama needs no authentication headers."""
        return {"Content-Type": "application/json"}

    def build_request(self, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        """Request body; JSON mode is requested through the format field."""
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }
        if json_mode:
            body["format"] = "json"
        return body

    def parse_response(self, payload: dict[str, Any]) -> ProviderResponse:
        """Extract the message content and token counts."""
        message = payload.get("message") or {}
        return ProviderResponse(
            content=message.get("content") or "",
            model=payload.get("model") or self._config.model,
            usage={
                "input_tokens": int(payload.get("prompt_eval_count", 0)),
                "output_tokens": int(payload.get("eval_count", 0)),
            },
        )


PROVIDERS: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def build_provider(config: AIConfig) -> BaseProvider:
    """Instantiate the provider named by ``config.provider``.

    Raises:
        ValueError: If the provider name is not supported.
    """
    provider_type = PROVIDERS.get(config.provider.lower())
    if provider_type is None:
        supported = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown AI provider {config.provider!r}. Supported: {supported}")
    return provider_type(config)