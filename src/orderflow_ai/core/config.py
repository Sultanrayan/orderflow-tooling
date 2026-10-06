"""Typed, validated configuration for the whole engine.

A configuration is a small tree of frozen dataclasses that mirrors the YAML
layout one to one. Loading is forgiving about unknown keys (they raise, so
typos never pass silently) and it merges a YAML file on top of the defaults,
which keeps config files short.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

from .timeframes import Timeframe, to_epoch_seconds

__all__ = [
    "ExchangeConfig",
    "PipelineConfig",
    "PrimitivesConfig",
    "PatternsConfig",
    "AIConfig",
    "OptimizationConfig",
    "OrderFlowConfig",
    "load_config",
]


@dataclass(slots=True, frozen=True)
class ExchangeConfig:
    """Which market to consume and how to shape the order book."""

    name: str = "binance"
    symbol: str = "BTCUSDT"
    tick_size: float = 1.0
    depth_levels: int = 20
    price_precision: int = 6
    base_url: str | None = None
    ws_url: str | None = None
    api_key: str | None = None
    api_secret: str | None = None

    @property
    def is_perpetual(self) -> bool:
        """``True`` for perpetual swap style symbols (USDT dated contracts)."""
        return self.symbol.upper().endswith(("PERP", "-PERP", "SWAP"))


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """Ingestion behaviour and lookback depth."""

    mode: str = "synthetic"
    timeframe: Timeframe = Timeframe.M1
    buffer_size: int = 10_000
    history_bars: int = 20
    record_path: str | None = None
    replay_path: str | None = None
    context_timeframes: tuple[str, ...] = ("5m", "15m", "1h")
    max_idle_seconds: float = 30.0
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe.from_string(str(self.timeframe)))
        object.__setattr__(self, "context_timeframes", tuple(self.context_timeframes))

    @property
    def is_live(self) -> bool:
        """``True`` when the pipeline consumes a live exchange feed."""
        return self.mode == "live"

    @property
    def replay_window(self) -> tuple[float | None, float | None]:
        """Inclusive start and exclusive end of the replay window, in epoch seconds.

        Returns:
            Two optional epoch-second bounds parsed from the ISO timestamps.
            ``(None, None)`` when no window is configured.
        """
        return _parse_time(self.start), _parse_time(self.end)


@dataclass(slots=True, frozen=True)
class PrimitivesConfig:
    """Thresholds used by the layer 2 primitives."""

    delta_ratio_threshold: float = 0.15
    imbalance_ratio_threshold: float = 1.5
    absorption_min_confidence: float = 60.0


@dataclass(slots=True, frozen=True)
class PatternsConfig:
    """Detection and filtering rules for layer 3."""

    min_confidence: float = 50.0
    max_patterns: int = 5
    max_age_bars: int = 3


@dataclass(slots=True, frozen=True)
class AIConfig:
    """Model provider settings for the layer 5 to AI handoff."""

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 300
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 30.0
    min_reward_risk: float = 1.5

    @property
    def resolved_base_url(self) -> str:
        """Base URL for the provider, falling back to the environment then defaults."""
        from_env = os.environ.get(f"{self.provider.upper()}_BASE_URL")
        return self.base_url or from_env or _DEFAULT_BASE_URLS.get(self.provider.lower(), "")

    @property
    def resolved_api_key(self) -> str | None:
        """API key for the provider, falling back to the environment."""
        return self.api_key or os.environ.get(f"{self.provider.upper()}_API_KEY")

    @property
    def requires_api_key(self) -> bool:
        """``True`` when the provider needs credentials (Ollama does not)."""
        return self.provider.lower() != "ollama"


_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://localhost:11434",
}


@dataclass(slots=True, frozen=True)
class OptimizationConfig:
    """Token budget switches used by layer 5."""

    tier: str = "auto"
    compact_json: bool = True
    short_keys: bool = True
    drop_nulls: bool = True
    round_numbers: bool = True
    delta_encode: bool = True
    cache_static: bool = True
    max_patterns: int = 3
    max_levels: int = 3
    price_decimals: int = 4
    ratio_decimals: int = 3


@dataclass(slots=True, frozen=True)
class OrderFlowConfig:
    """Root configuration object passed through every layer."""

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    primitives: PrimitivesConfig = field(default_factory=PrimitivesConfig)
    patterns: PatternsConfig = field(default_factory=PatternsConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @property
    def symbol(self) -> str:
        """Shortcut for the traded symbol."""
        return self.exchange.symbol

    @property
    def timeframe(self) -> Timeframe:
        """Shortcut for the base candle timeframe."""
        return self.pipeline.timeframe

    def analysis_timeframes(self) -> tuple[Timeframe, ...]:
        """Base timeframe plus the longer context timeframes, de-duplicated."""
        seen: dict[Timeframe, None] = {}
        for label in (self.timeframe.value, *self.pipeline.context_timeframes):
            timeframe = Timeframe.from_string(label)
            if timeframe.seconds >= self.timeframe.seconds:
                seen[timeframe] = None
        return tuple(sorted(seen, key=lambda item: item.seconds))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> OrderFlowConfig:
        """Build a config from a nested mapping, filling gaps with defaults.

        Args:
            data: Parsed YAML document, or ``None`` for pure defaults.

        Returns:
            A fully populated configuration.

        Raises:
            TypeError: If a leaf value cannot be coerced to the declared type.
            ValueError: If an unknown configuration key is present.
        """
        data = data or {}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
        return cls(
            exchange=_build(ExchangeConfig, data.get("exchange")),
            pipeline=_build(PipelineConfig, data.get("pipeline")),
            primitives=_build(PrimitivesConfig, data.get("primitives")),
            patterns=_build(PatternsConfig, data.get("patterns")),
            ai=_build(AIConfig, data.get("ai")),
            optimization=_build(OptimizationConfig, data.get("optimization")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> OrderFlowConfig:
        """Load a configuration from a YAML file."""
        return cls.from_mapping(_read_yaml(Path(path)))

    @classmethod
    def load(cls, source: str | Path | OrderFlowConfig | Mapping[str, Any] | None) -> OrderFlowConfig:
        """Resolve a config from a path, mapping, instance or ``None``."""
        if isinstance(source, OrderFlowConfig):
            return source
        if isinstance(source, Mapping):
            return cls.from_mapping(source)
        if source is None:
            return cls.from_mapping(None)
        return cls.from_yaml(Path(str(source)))


def load_config(source: str | Path | OrderFlowConfig | Mapping[str, Any] | None = None) -> OrderFlowConfig:
    """Module level shortcut for :meth:`OrderFlowConfig.load`."""
    return OrderFlowConfig.load(source)


def _build(dataclass_type: type, data: Mapping[str, Any] | None) -> Any:
    """Instantiate ``dataclass_type`` from ``data`` honouring declared types."""
    if data is None:
        return dataclass_type()
    if not isinstance(data, Mapping):
        raise TypeError(
            f"Section {dataclass_type.__name__} must be a mapping, got {type(data).__name__}"
        )

    annotations = _resolved_hints(dataclass_type)
    known = {f.name for f in fields(dataclass_type)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys in {dataclass_type.__name__}: {', '.join(sorted(unknown))}")

    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        kwargs[key] = _coerce(value, _unwrap_optional(annotations.get(key)))
    return dataclass_type(**kwargs)


@cache
def _resolved_hints(dataclass_type: type) -> dict[str, Any]:
    """Return resolved type hints for a config dataclass.

    ``from __future__ import annotations`` turns every annotation into a
    string, so hints are resolved once and memoised.
    """
    return get_type_hints(dataclass_type)


def _coerce(value: Any, target_type: Any) -> Any:
    """Convert ``value`` to ``target_type`` where the intent is unambiguous."""
    if target_type is None or target_type is Any:
        return value
    if target_type is Timeframe:
        return Timeframe.from_string(str(value))
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    if is_dataclass(target_type):
        return _build(target_type, value)
    if target_type is tuple or get_origin(target_type) is tuple:
        return tuple(value)
    return value


def _unwrap_optional(annotation: Any) -> Any:
    """Return ``X`` for ``X | None`` annotations."""
    if get_origin(annotation) in (Union, UnionType):
        return next(arg for arg in get_args(annotation) if arg is not NoneType)
    return annotation


def _parse_time(value: str | None) -> float | None:
    """Parse an ISO 8601 timestamp into epoch seconds."""
    if not value:
        return None
    return to_epoch_seconds(datetime.fromisoformat(value))


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Configuration root must be a mapping, got {type(loaded).__name__}")
    return loaded