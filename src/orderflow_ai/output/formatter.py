"""Payload formatting.

Formatting happens in two steps. The formatter first asks the summariser for a
verbose payload: complete, readable and useful on its own for debugging. It then
applies array encoding, which shortens the section names and replaces every
object with known keys by a fixed position list, and hands the result to the
optimiser for the remaining token strategies. Both steps are pure functions of
the analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.analysis import Analysis
from .optimizer import OptimizationReport, PayloadOptimizer, dumps, estimate_tokens
from .prioritizer import Prioritisation, Prioritizer
from .summarizer import Summarizer
from .tiers import TIER_FULL, TIER_MINIMAL, TIER_STANDARD, TierSelector

__all__ = ["PayloadBuild", "PayloadBuilder", "SECTION_KEYS", "encode_arrays"]

MAX_FOOTPRINT_LEVELS = 4
DEFAULT_PRICE_DECIMALS = 4
DEFAULT_RATIO_DECIMALS = 3

# Sections included per tier, ordered from cheapest to most expensive.
TIER_SECTIONS: dict[str, tuple[str, ...]] = {
    TIER_MINIMAL: ("market", "direction", "patterns", "levels", "context"),
    TIER_STANDARD: (
        "market",
        "direction",
        "patterns",
        "levels",
        "context",
        "structure",
        "pressure",
        "quality",
    ),
    TIER_FULL: (
        "market",
        "direction",
        "patterns",
        "levels",
        "context",
        "structure",
        "pressure",
        "quality",
        "fallback",
    ),
}

# Section name -> short key, matching the legend sent with the prompt.
SECTION_KEYS: dict[str, str] = {
    "market": "m",
    "direction": "d",
    "patterns": "pat",
    "levels": "lvl",
    "context": "c",
    "structure": "s",
    "pressure": "p",
    "quality": "q",
    "fallback": "fb",
}


@dataclass(frozen=True, slots=True)
class PayloadBuild:
    """One formatted and optimised payload."""

    payload: dict[str, Any]
    report: OptimizationReport
    priority: Prioritisation
    tier: str

    @property
    def token_estimate(self) -> int:
        """Estimated tokens of the serialised payload."""
        return estimate_tokens(dumps(self.payload))


class PayloadBuilder:
    """Builds compact payloads for the AI stage."""

    def __init__(
        self,
        optimizer: PayloadOptimizer | None = None,
        summarizer: Summarizer | None = None,
        tier_selector: TierSelector | None = None,
        max_footprint_levels: int = MAX_FOOTPRINT_LEVELS,
    ) -> None:
        """Create the builder.

        Args:
            optimizer: Token optimiser. Defaults to all strategies enabled.
            summarizer: Evidence summariser.
            tier_selector: Tier selection policy.
            max_footprint_levels: Footprint levels shipped per bar.
        """
        self._summarizer = summarizer or Summarizer()
        self._prioritizer = Prioritizer(self._summarizer)
        self._optimizer = optimizer or PayloadOptimizer(
            price_decimals=DEFAULT_PRICE_DECIMALS, ratio_decimals=DEFAULT_RATIO_DECIMALS
        )
        self._tiers = tier_selector or TierSelector()
        self._max_footprint_levels = max_footprint_levels

    @property
    def optimizer(self) -> PayloadOptimizer:
        """Optimiser in use, exposed for token budgeting and tests."""
        return self._optimizer

    @property
    def summarizer(self) -> Summarizer:
        """Summariser in use."""
        return self._summarizer

    def tier_of(self, analysis: Analysis) -> str:
        """Tier that would be selected for ``analysis``."""
        return self._tiers.select(analysis).tier

    def build(self, analysis: Analysis, tier: str | None = None) -> dict[str, Any]:
        """Build the verbose payload with full key names.

        Args:
            analysis: The analysis to encode.
            tier: Force a tier instead of selecting one.

        Returns:
            A nested dictionary holding only the sections allowed by the tier.
        """
        chosen = tier or self._tiers.select(analysis).tier
        available = {
            "market": {**self._summarizer.market(analysis), "tier": chosen},
            "direction": self._summarizer.direction(analysis.primitives),
            "patterns": self._summarizer.patterns(analysis.patterns.patterns),
            "levels": self._summarizer.levels(analysis.context),
            "context": self._summarizer.context(analysis.context),
            "structure": self._summarizer.structure(
                analysis.primitives, self._max_footprint_levels
            ),
            "pressure": self._summarizer.pressure(analysis.primitives),
            "quality": self._summarizer.quality(analysis.primitives),
            "fallback": self._summarizer.fallback(analysis.patterns),
        }
        sections = TIER_SECTIONS.get(chosen, TIER_SECTIONS[TIER_STANDARD])
        return {
            name: available[name]
            for name in sections
            if available.get(name) is not None
        }

    def build_optimized(self, analysis: Analysis, tier: str | None = None) -> PayloadBuild:
        """Build, array encode, optimise and report in one call.

        Returns:
            The optimised payload together with its size report, the chosen tier
            and the prioritisation that decided the leading evidence. The report
            measures the reduction against the verbose payload, which is the
            honest "before" number.
        """
        chosen = tier or self._tiers.select(analysis).tier
        priority = self._prioritizer.prioritise(analysis, chosen)
        verbose = self.build(analysis, chosen)
        payload, report = self._optimizer.optimize(
            encode_arrays(verbose, self._optimizer.price_decimals, self._optimizer.ratio_decimals),
            baseline=verbose,
        )
        return PayloadBuild(payload=payload, report=report, priority=priority, tier=chosen)

    def prioritise(self, analysis: Analysis, tier: str | None = None) -> Prioritisation:
        """Rank the evidence without building a payload."""
        chosen = tier or self._tiers.select(analysis).tier
        return self._prioritizer.prioritise(analysis, chosen)

    def serialize(self, analysis: Analysis, tier: str | None = None) -> str:
        """Build, optimise and serialise in one call."""
        return self._optimizer.serialize(self.build_optimized(analysis, tier).payload)


def encode_arrays(
    payload: dict[str, Any],
    price_decimals: int = DEFAULT_PRICE_DECIMALS,
    ratio_decimals: int = DEFAULT_RATIO_DECIMALS,
) -> dict[str, Any]:
    """Shorten section names and replace known shapes by fixed position lists.

    ``{"poc": 1.0852, "vah": 1.0854, "val": 1.0851}`` becomes
    ``[1.0852, 1.0854, 1.0851]``. The prompt legend documents every position, and
    the saving is worth roughly a third of the section size.

    Values are rounded here rather than by the optimiser, because only this step
    knows which position of a row holds a price. Sections without a documented
    shape are returned unchanged.

    Args:
        payload: Verbose payload as produced by :meth:`PayloadBuilder.build`.
        price_decimals: Decimals kept for prices.
        ratio_decimals: Decimals kept for ratios and scores.

    Returns:
        The array encoded payload with short section keys.
    """
    encoded: dict[str, Any] = {}
    for name, section in payload.items():
        encoder = _ENCODERS.get(name)
        value = encoder(section, price_decimals, ratio_decimals) if encoder else section
        encoded[SECTION_KEYS.get(name, name)] = value
    return encoded


def _price(value: Any, decimals: int) -> Any:
    """Round a price, leaving anything that is not a number untouched."""
    return round(value, decimals) if isinstance(value, (int, float)) else value


def _ratio(value: Any, decimals: int) -> Any:
    """Round a ratio or a score, leaving anything that is not a number alone."""
    return round(value, decimals) if isinstance(value, (int, float)) else value


def _encode_market(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[symbol, timeframe, bar open, price, tier]``."""
    return [
        section.get("symbol"),
        section.get("timeframe"),
        section.get("bar_open"),
        _price(section.get("price"), pd),
        section.get("tier"),
    ]


def _encode_direction(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[delta, delta ratio, direction, cvd, cvd slope, momentum, divergence]``.

    An absent divergence is encoded as an empty string so the positions of this
    array never shift.
    """
    return [
        _ratio(section.get("delta"), rd),
        _ratio(section.get("delta_ratio"), rd),
        section.get("direction"),
        _ratio(section.get("cvd"), rd),
        _ratio(section.get("cvd_slope"), rd),
        section.get("momentum"),
        section.get("divergence") or "",
    ]


def _encode_patterns(section: list[dict[str, Any]], pd: int, rd: int) -> list[list[Any]]:
    """``[[name, price, direction, confidence, bars ago], ...]``."""
    return [
        [
            item.get("name"),
            _price(item.get("price"), pd),
            item.get("direction"),
            _ratio(item.get("confidence"), 0),
            item.get("bars_ago"),
        ]
        for item in section
    ]


def _encode_levels(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[support, resistance, point of control]``. Zero means "no level"."""
    return [
        [_price(level, pd) for level in section.get("support") or []],
        [_price(level, pd) for level in section.get("resistance") or []],
        _price(section.get("poc") or 0.0, pd),
    ]


def _encode_context(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[structure, bos, choch, session, volatility regime, mtf score]``."""
    volatility = section.get("volatility") or {}
    multi_timeframe = section.get("multi_timeframe") or {}
    return [
        section.get("structure"),
        section.get("bos"),
        bool(section.get("choch")),
        section.get("session"),
        volatility.get("regime"),
        _ratio(multi_timeframe.get("score"), 0),
    ]


def _encode_structure(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[ohlc, volume, delta, levels, profile, migration]``."""
    footprint = section.get("footprint") or {}
    profile = section.get("volume_profile") or {}
    migration = section.get("value_migration") or {}
    levels = [
        [
            _price(row.get(key), pd) if key == "price" else _ratio(row.get(key), rd)
            for key in ("price", "bid", "ask")
        ]
        for row in footprint.get("levels") or []
    ]
    volume_profile = [_price(profile.get(key), pd) for key in ("poc", "vah", "val")]
    volume_profile.append([_price(level, pd) for level in profile.get("hvn") or []])
    volume_profile.append([_price(level, pd) for level in profile.get("lvn") or []])
    return [
        [_price(footprint.get(key), pd) for key in ("open", "high", "low", "close")],
        _ratio(footprint.get("volume"), rd),
        _ratio(footprint.get("delta"), rd),
        levels,
        volume_profile,
        [migration.get("direction"), _ratio(migration.get("strength"), rd)],
    ]


def _encode_pressure(section: dict[str, Any], pd: int, rd: int) -> dict[str, Any]:
    """``ob`` holds book depth, ``top`` the strongest footprint imbalance.

    ``top`` is an empty list when no level was imbalanced, which lets the
    optimiser drop the key entirely.
    """
    orderbook = section.get("orderbook") or {}
    imbalance = section.get("imbalance") or {}
    strongest = imbalance.get("strongest") or {}
    return {
        "ob": [
            _ratio(orderbook.get("bid_depth"), rd),
            _ratio(orderbook.get("ask_depth"), rd),
            _ratio(orderbook.get("imbalance_ratio"), rd),
            orderbook.get("direction") or "balanced",
        ],
        "top": [
            _price(strongest.get("price"), pd),
            strongest.get("side"),
            _ratio(strongest.get("ratio"), rd),
        ]
        if strongest
        else [],
    }


def _encode_quality(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[[intensity ratio, percentile, level], [absorption flag, confidence, side]]``."""
    intensity = section.get("volume_intensity") or {}
    absorption = section.get("absorption") or {}
    return [
        [
            _ratio(intensity.get("ratio"), rd),
            _ratio(intensity.get("percentile"), 0),
            intensity.get("level"),
        ],
        [
            bool(absorption.get("detected")),
            _ratio(absorption.get("confidence"), 0),
            absorption.get("side"),
        ],
    ]


def _encode_fallback(section: dict[str, Any], pd: int, rd: int) -> list[Any]:
    """``[score, tier, [[component, points], ...]]``."""
    return [
        _ratio(section.get("score"), 0),
        section.get("tier"),
        [
            [item.get("name"), _ratio(item.get("points"), 1)]
            for item in section.get("components") or []
        ],
    ]


_ENCODERS = {
    "market": _encode_market,
    "direction": _encode_direction,
    "patterns": _encode_patterns,
    "levels": _encode_levels,
    "context": _encode_context,
    "structure": _encode_structure,
    "pressure": _encode_pressure,
    "quality": _encode_quality,
    "fallback": _encode_fallback,
}