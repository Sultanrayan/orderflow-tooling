"""Token optimisation strategies.

The strategies from the specification are implemented here as independent,
individually switchable passes over the payload:

===========================  ==============================================
Strategy                     Implementation
===========================  ==============================================
Compact JSON                 separators without spaces
Array encoding               objects with known keys become lists
Short keys                   per section key map
Remove nulls                 empty sections and ``None`` values are dropped
Round numbers                prices and ratios are rounded
Delta encoding               numeric series become ``[first, diff, diff…]``
Summary instead of raw       the summariser already drops noise
Cache static parts           the static prompt block is hashed by
                             :mod:`orderflow_ai.output.prompt`
Tiered complexity            sections per tier, see
                             :mod:`orderflow_ai.output.tiers`
===========================  ==============================================

Scalar rounding works on keys: a float stored under a key listed in
:data:`PRICE_KEYS` or :data:`RATIO_KEYS` is rounded, everything else is left
alone. Values inside arrays are rounded by the formatter instead, because only
the formatter knows which element of a row is a price.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OptimizationReport", "PayloadOptimizer", "dumps", "estimate_tokens"]

CHARS_PER_TOKEN = 4

# Verbose key -> short key, applied recursively to every object.
KEY_MAP: dict[str, str] = {
    "symbol": "s",
    "timeframe": "tf",
    "bar_open": "t",
    "price": "px",
    "tier": "tr",
    "delta": "d",
    "delta_ratio": "dr",
    "direction": "dir",
    "delta_strength": "ds",
    "cvd": "cv",
    "cvd_slope": "cs",
    "momentum": "mo",
    "divergence": "dv",
    "divergence_confidence": "dc",
    "open": "o",
    "high": "h",
    "low": "l",
    "close": "c",
    "volume": "v",
    "trades": "t",
    "levels": "lv",
    "footprint": "fp",
    "volume_profile": "vp",
    "poc": "poc",
    "vah": "vah",
    "val": "val",
    "hvn": "hvn",
    "lvn": "lvn",
    "value_migration": "vm",
    "strength": "st",
    "imbalance": "im",
    "threshold": "th",
    "count": "n",
    "strongest": "top",
    "side": "sd",
    "ratio": "r",
    "orderbook": "ob",
    "bid_depth": "bd",
    "ask_depth": "ad",
    "spread_bps": "sp",
    "walls": "wl",
    "size": "sz",
    "volume_intensity": "vi",
    "percentile": "pc",
    "level": "lv",
    "absorption": "ab",
    "detected": "dt",
    "confidence": "cf",
    "structure": "st",
    "bos": "bos",
    "choch": "ch",
    "session": "ss",
    "active_sessions": "as",
    "volatility": "vol",
    "regime": "rg",
    "average_range": "ar",
    "multi_timeframe": "mtf",
    "score": "sc",
    "level_proximity": "lp",
    "support": "sup",
    "resistance": "res",
    "patterns": "pat",
    "name": "nm",
    "label": "lb",
    "bars_ago": "ba",
    "evidence": "ev",
    "fallback": "fb",
    "components": "cp",
    "points": "pt",
    "max": "mx",
}

# Keys holding prices and keys holding ratios or scores.
PRICE_KEYS = frozenset(
    {"px", "poc", "vah", "val", "hvn", "lvn", "sup", "res", "o", "h", "l", "c"}
)
RATIO_KEYS = frozenset({"r", "dr", "ds", "cs", "st", "lp", "ar", "sp", "pt", "pc", "sc", "cf"})
# Lists of numbers that are worth delta encoding.
DELTA_SERIES = frozenset({"ev"})


@dataclass(slots=True)
class OptimizationReport:
    """Size accounting for one optimisation run."""

    raw_bytes: int
    final_bytes: int
    strategies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def saved_bytes(self) -> int:
        """Bytes removed compared with the raw payload."""
        return max(0, self.raw_bytes - self.final_bytes)

    @property
    def reduction(self) -> float:
        """Fraction of bytes removed, ``0.0`` to ``1.0``."""
        if self.raw_bytes <= 0:
            return 0.0
        return self.saved_bytes / self.raw_bytes

    def summary(self) -> str:
        """One line human readable summary."""
        return (
            f"{self.raw_bytes}B -> {self.final_bytes}B "
            f"({self.reduction * 100:.0f}% smaller, {len(self.strategies)} strategies)"
        )


class PayloadOptimizer:
    """Applies the token optimisation passes to a payload."""

    def __init__(
        self,
        compact_json: bool = True,
        short_keys: bool = True,
        drop_nulls: bool = True,
        round_numbers: bool = True,
        delta_encode: bool = True,
        price_decimals: int = 4,
        ratio_decimals: int = 3,
    ) -> None:
        """Create the optimiser.

        Args:
            compact_json: Serialise without whitespace.
            short_keys: Replace verbose keys with the documented short forms.
            drop_nulls: Remove ``None`` values and empty containers.
            round_numbers: Round prices and ratios before serialising.
            delta_encode: Delta encode the listed numeric series.
            price_decimals: Decimals kept for prices.
            ratio_decimals: Decimals kept for ratios and scores.
        """
        self.compact_json = compact_json
        self.short_keys = short_keys
        self.drop_nulls = drop_nulls
        self.round_numbers = round_numbers
        self.delta_encode = delta_encode
        self.price_decimals = price_decimals
        self.ratio_decimals = ratio_decimals

    def optimize(
        self, payload: dict[str, Any], baseline: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], OptimizationReport]:
        """Optimise ``payload`` and report the size change.

        Args:
            payload: Payload as produced by the formatter.
            baseline: Payload to measure against. Defaults to ``payload``; pass
                the verbose payload to report the reduction including array
                encoding.

        Returns:
            The optimised payload and an :class:`OptimizationReport`.
        """
        raw_bytes = len(dumps(baseline if baseline is not None else payload, compact=False))
        applied: list[str] = []
        result: Any = payload

        if self.short_keys:
            result = _shorten_keys(result)
            applied.append("short_keys")
        if self.drop_nulls:
            result = _drop_empty(result)
            applied.append("drop_nulls")
        if self.round_numbers:
            result = _round_scalars(result, self.price_decimals, self.ratio_decimals)
            applied.append("round_numbers")
        if self.delta_encode:
            result = _delta_encode(result)
            applied.append("delta_encode")

        final_bytes = len(dumps(result, compact=self.compact_json))
        if self.compact_json:
            applied.append("compact_json")

        return result, OptimizationReport(
            raw_bytes=raw_bytes, final_bytes=final_bytes, strategies=tuple(applied)
        )

    def serialize(self, payload: Any) -> str:
        """Serialise ``payload`` using the configured JSON style."""
        return dumps(payload, compact=self.compact_json)

    def array_encode(self, rows: Iterable[Sequence[Any]]) -> list[list[Any]]:
        """Encode rows of values as plain lists, which JSON always supports."""
        return [list(row) for row in rows]


def dumps(payload: Any, compact: bool = True) -> str:
    """Serialise ``payload`` to JSON, optionally without whitespace."""
    separators = (",", ":") if compact else (", ", ": ")
    return json.dumps(payload, separators=separators, ensure_ascii=False, default=str)


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Uses the widely applied four characters per token rule, which is accurate
    enough for budgeting and needs no tokenizer dependency.
    """
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _shorten_keys(value: Any) -> Any:
    """Recursively rename keys using :data:`KEY_MAP`."""
    if isinstance(value, dict):
        return {KEY_MAP.get(key, key): _shorten_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shorten_keys(item) for item in value]
    return value


def _drop_empty(value: Any) -> Any:
    """Recursively drop ``None`` values and empty containers."""
    if isinstance(value, dict):
        return {
            key: item
            for key, item in ((key, _drop_empty(item)) for key, item in value.items())
            if item not in (None, {}, [])
        }
    if isinstance(value, list):
        return [item for item in (_drop_empty(entry) for entry in value) if item not in (None, {}, [])]
    return value


def _round_scalars(value: Any, price_decimals: int, ratio_decimals: int, key: str | None = None) -> Any:
    """Round floats whose key is registered as a price or a ratio."""
    if isinstance(value, dict):
        return {
            name: _round_scalars(item, price_decimals, ratio_decimals, key=name)
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [_round_scalars(item, price_decimals, ratio_decimals, key=key) for item in value]
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    if key in PRICE_KEYS:
        return round(value, price_decimals)
    if key in RATIO_KEYS:
        return round(value, ratio_decimals)
    return value


def _delta_encode(value: Any) -> Any:
    """Delta encode the registered numeric series."""
    if isinstance(value, dict):
        return {
            key: _encode_series(item) if key in DELTA_SERIES else _delta_encode(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_delta_encode(item) for item in value]
    return value


def _encode_series(value: Any) -> Any:
    """Encode ``[a, b, c]`` as ``[a, b-a, c-b]``."""
    if not isinstance(value, list) or len(value) < 2:
        return value
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return value
    encoded: list[float] = [value[0]]
    for previous, current in zip(value, value[1:], strict=False):
        encoded.append(round(current - previous, 6))
    return encoded