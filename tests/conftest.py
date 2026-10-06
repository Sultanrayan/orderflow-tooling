"""Shared pytest fixtures.

The suite never touches the network: layer 1 is exercised with the synthetic
feed and with recorded JSON lines, and the AI stage is exercised with a stub
provider.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from orderflow_ai.ai.providers import BaseProvider, ProviderResponse
from orderflow_ai.core.analysis import Analysis
from orderflow_ai.core.config import AIConfig, OrderFlowConfig
from orderflow_ai.ingestion.feeds.synthetic import SyntheticFeed
from orderflow_ai.ingestion.recorder import TickRecorder
from tests.factories import make_bar, make_book, make_config


def run_async(coroutine: Any) -> Any:
    """Run a coroutine to completion, for use inside synchronous fixtures."""
    return asyncio.run(coroutine)


@pytest.fixture
def config() -> OrderFlowConfig:
    """A synthetic-mode configuration tuned for fast tests."""
    return make_config()


@pytest.fixture
def bar():
    """Factory building a single footprint bar."""
    return make_bar


@pytest.fixture
def book():
    """Factory building an order book snapshot."""
    return make_book


@pytest.fixture
def analysis(config: OrderFlowConfig) -> Analysis:
    """A fully built analysis produced by the real pipeline."""
    from orderflow_ai.core.pipeline import OrderFlowPipeline

    feed = SyntheticFeed(config.exchange, config.timeframe, bars=30, seed=11)
    pipeline = OrderFlowPipeline(config, trade_feed=feed)
    analyses = run_async(pipeline.collect(limit=12))
    assert analyses, "synthetic feed produced no bars"
    return analyses[-1]


@pytest.fixture
def recorded_file(tmp_path: Path) -> Path:
    """A short recorded session on disk, ready for the replay feed."""
    path = tmp_path / "ticks.jsonl"
    feed = SyntheticFeed(bars=6, seed=5)

    async def record() -> None:
        trades = []
        async for trade in feed.stream():
            trades.append(trade)
        with TickRecorder(path) as recorder:
            for trade in trades:
                recorder.record_trade(trade)
            recorder.record_book(feed.snapshot(), trades[-1].ts, force=True)

    run_async(record())
    return path


def default_reply() -> dict:
    """A schema-valid, high confidence response used across the tests."""
    return {
        "bias": "bullish",
        "confidence": 78,
        "signal": "BUY",
        "primary_evidence": "patterns",
        "reasoning": "Strong absorption at the point of control.",
        "key_levels": {"support": [99.5], "resistance": [101.0]},
        "setup": {"entry": 100.0, "stop": 99.0, "target": 102.0, "rr": 2.0},
        "wait_for": None,
        "invalidation": "Close below 99.0",
        "risk_notes": "R:R 2.0 acceptable.",
    }


def reply_for(analysis, **overrides) -> dict:
    """A valid reply whose reasoning only cites patterns present in ``analysis``.

    The validator rejects reasoning that invents patterns, so tests that want an
    accepted signal must build the reasoning from the analysis itself.
    """
    patterns = list(analysis.pattern_names)
    reasoning = (
        f"Evidence from {', '.join(patterns)} at the point of control."
        if patterns
        else "No pattern detected, primitives stay neutral."
    )
    return default_reply() | {"reasoning": reasoning} | overrides


@dataclass
class StubProvider(BaseProvider):
    """Test double returning a canned completion payload without transport."""

    reply: dict | str = field(default_factory=default_reply)
    requests: list[dict] = field(default_factory=list)

    name: str = "stub"

    def __post_init__(self) -> None:
        super().__init__(AIConfig(provider="stub", model="stub"))

    def endpoint(self) -> str:
        """No endpoint is used."""
        return "memory://stub"

    def headers(self) -> dict[str, str]:
        """No headers are needed."""
        return {}

    def build_request(self, messages: list[dict[str, str]], json_mode: bool) -> dict:
        """Record the request; the body is never sent anywhere."""
        self.requests.append({"messages": messages, "json_mode": json_mode})
        return {}

    def parse_response(self, payload: dict) -> ProviderResponse:
        """Return the canned reply."""
        content = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return ProviderResponse(content=content, model="stub")

    def requires_api_key(self) -> bool:
        """The stub needs no credentials."""
        return False

    async def complete(self, messages, json_mode: bool = True, client=None) -> ProviderResponse:
        """Return the canned reply without touching the network."""
        self.build_request(messages, json_mode)
        return self.parse_response({})


@pytest.fixture
def stub_provider() -> StubProvider:
    """A provider that returns a canned JSON reply."""
    return StubProvider()