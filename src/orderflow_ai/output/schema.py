"""Response schema and strict validation of the model reply.

The specification requires four gates before a signal is accepted: the schema
must hold, the confidence must match the signal, the reward to risk must clear
the minimum, and the model must not have invented a pattern. Everything here is
pure validation on already parsed JSON, with no network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..patterns.engine import default_detectors

__all__ = [
    "BIASES",
    "EVIDENCE_SOURCES",
    "SIGNALS",
    "Signal",
    "SignalError",
    "SignalSchema",
    "SignalValidator",
    "TradingSignal",
    "ValidationResult",
]


class Bias(str, Enum):
    """Directional view of the market."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Signal(str, Enum):
    """Action the model recommends."""

    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"
    WAIT = "WAIT"


class Evidence(str, Enum):
    """Which part of the payload the decision leans on."""

    PATTERNS = "patterns"
    PRIMITIVES = "primitives"
    CONTEXT = "context"


BIASES = tuple(bias.value for bias in Bias)
SIGNALS = tuple(signal.value for signal in Signal)
EVIDENCE_SOURCES = tuple(source.value for source in Evidence)

# Directional signals must clear this confidence; anything lower must wait.
MIN_DIRECTIONAL_CONFIDENCE = 50
DIRECTIONAL_SIGNALS = frozenset({Signal.BUY, Signal.SELL})
DIRECTIONAL_TRUST = 75.0
# Fields every response must carry. ``setup`` is deliberately absent: it is
# meaningless for ``NO_TRADE`` and ``WAIT``.
REQUIRED_FIELDS = (
    "bias",
    "confidence",
    "signal",
    "primary_evidence",
    "reasoning",
)


class SignalError(ValueError):
    """Raised when a response cannot be turned into a :class:`TradingSignal`."""


@dataclass(frozen=True, slots=True)
class TradingSignal:
    """A validated model decision."""

    bias: str
    confidence: float
    signal: str
    primary_evidence: str
    reasoning: str
    key_levels: dict[str, list[float]] = field(default_factory=dict)
    setup: dict[str, float] | None = None
    wait_for: str | None = None
    invalidation: str = ""
    risk_notes: str = ""

    @property
    def is_actionable(self) -> bool:
        """``True`` for ``BUY`` and ``SELL``."""
        return self.signal in {Signal.BUY.value, Signal.SELL.value}

    @property
    def reward_risk(self) -> float | None:
        """Reward to risk of the proposed setup, if one was given."""
        return self.setup.get("rr") if self.setup else None

    def to_dict(self) -> dict[str, Any]:
        """JSON friendly view of the signal."""
        return {
            "bias": self.bias,
            "confidence": self.confidence,
            "signal": self.signal,
            "primary_evidence": self.primary_evidence,
            "reasoning": self.reasoning,
            "key_levels": self.key_levels,
            "setup": self.setup,
            "wait_for": self.wait_for,
            "invalidation": self.invalidation,
            "risk_notes": self.risk_notes,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating a model response."""

    valid: bool
    signal: TradingSignal | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.valid

    def raise_for_errors(self) -> TradingSignal:
        """Return the signal or raise :class:`SignalError` with all errors."""
        if self.signal is None:
            raise SignalError("; ".join(self.errors) or "invalid response")
        return self.signal


class SignalSchema:
    """The JSON schema the model is asked to answer with."""

    @property
    def json_schema(self) -> dict[str, Any]:
        """A JSON schema object describing a valid response."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(REQUIRED_FIELDS),
            "properties": {
                "bias": {"type": "string", "enum": list(BIASES)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                "signal": {"type": "string", "enum": list(SIGNALS)},
                "primary_evidence": {"type": "string", "enum": list(EVIDENCE_SOURCES)},
                "reasoning": {"type": "string", "minLength": 1},
                "key_levels": {
                    "type": "object",
                    "properties": {
                        "support": {"type": "array", "items": {"type": "number"}},
                        "resistance": {"type": "array", "items": {"type": "number"}},
                    },
                },
                "setup": {
                    "type": "object",
                    "required": ["entry", "stop", "target"],
                    "properties": {
                        "entry": {"type": "number"},
                        "stop": {"type": "number"},
                        "target": {"type": "number"},
                        "rr": {"type": "number"},
                    },
                },
                "wait_for": {"type": ["string", "null"]},
                "invalidation": {"type": "string"},
                "risk_notes": {"type": "string"},
            },
        }

    def validate_shape(self, response: dict[str, Any]) -> list[str]:
        """Check required fields, types and enum values.

        Args:
            response: Decoded model response.

        Returns:
            One message per problem found. Empty when the shape is valid.
        """
        errors: list[str] = []
        if not isinstance(response, dict):
            return ["response must be a JSON object"]

        for name in REQUIRED_FIELDS:
            if name not in response:
                errors.append(f"missing required field '{name}'")

        bias = response.get("bias")
        if bias is not None and bias not in BIASES:
            errors.append(f"bias must be one of {BIASES}, got {bias!r}")

        signal = response.get("signal")
        if signal is not None and signal not in SIGNALS:
            errors.append(f"signal must be one of {SIGNALS}, got {signal!r}")

        evidence = response.get("primary_evidence")
        if evidence is not None and evidence not in EVIDENCE_SOURCES:
            errors.append(f"primary_evidence must be one of {EVIDENCE_SOURCES}, got {evidence!r}")

        confidence = response.get("confidence")
        if confidence is not None and not _is_number(confidence):
            errors.append("confidence must be a number")
        elif isinstance(confidence, (int, float)) and not 0 <= confidence <= 100:
            errors.append("confidence must be between 0 and 100")

        if "reasoning" in response and not str(response["reasoning"]).strip():
            errors.append("reasoning must not be empty")

        setup = response.get("setup")
        if setup is not None:
            errors.extend(_setup_errors(setup))

        return errors


class SignalValidator:
    """Applies the four documented acceptance gates."""

    def __init__(
        self,
        schema: SignalSchema | None = None,
        min_reward_risk: float = 1.5,
        min_confidence: float = MIN_DIRECTIONAL_CONFIDENCE,
        known_patterns: tuple[str, ...] = (),
    ) -> None:
        """Create the validator.

        Args:
            schema: Response schema. A default one is created when omitted.
            min_reward_risk: Minimum acceptable reward to risk.
            min_confidence: Minimum confidence for a directional signal.
            known_patterns: Pattern names present in the payload. Used to catch
                hallucinated patterns in the reasoning.
        """
        self._schema = schema or SignalSchema()
        self._min_reward_risk = min_reward_risk
        self._min_confidence = min_confidence
        self._known_patterns = tuple(known_patterns)

    @property
    def min_reward_risk(self) -> float:
        """Minimum acceptable reward to risk."""
        return self._min_reward_risk

    def validate(
        self,
        response: dict[str, Any],
        known_patterns: tuple[str, ...] | None = None,
    ) -> ValidationResult:
        """Run every gate against ``response``.

        Args:
            response: Decoded model response.
            known_patterns: Pattern names present in the payload. Overrides the
                names given at construction time.

        Returns:
            The validation result. ``errors`` explains every rejection gate that
            failed; ``warnings`` records softer problems such as a missing
            setup on a directional signal.
        """
        errors = self._schema.validate_shape(response)
        if errors:
            return ValidationResult(valid=False, errors=tuple(errors))

        signal = _signal_of(response)
        patterns = self._known_patterns if known_patterns is None else tuple(known_patterns)
        errors.extend(self._confidence_errors(signal, response))
        errors.extend(self._reward_risk_errors(signal, response))
        errors.extend(self._hallucination_errors(response, patterns))
        warnings = tuple(self._soft_checks(signal, response))

        if errors:
            return ValidationResult(valid=False, errors=tuple(errors), warnings=warnings)
        return ValidationResult(valid=True, signal=signal, warnings=warnings)

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    def _confidence_errors(self, signal: TradingSignal, response: dict[str, Any]) -> list[str]:
        """A directional signal must be backed by matching bias and confidence."""
        errors: list[str] = []
        confidence = float(response.get("confidence", 0.0))
        bias = str(response.get("bias"))

        if signal.signal in DIRECTIONAL_SIGNALS:
            if confidence < self._min_confidence:
                errors.append(
                    f"{signal.signal} requires confidence >= {self._min_confidence}, got {confidence:.0f}"
                )
            expected = Bias.BULLISH.value if signal.signal == Signal.BUY.value else Bias.BEARISH.value
            if bias != expected:
                errors.append(f"{signal.signal} requires bias {expected}, got {bias}")
        elif signal.signal == Signal.NO_TRADE.value and confidence >= DIRECTIONAL_TRUST:
            errors.append(
                f"NO_TRADE contradicts confidence {confidence:.0f}; "
                "a directional signal was expected"
            )

        return errors

    def _reward_risk_errors(self, signal: TradingSignal, response: dict[str, Any]) -> list[str]:
        """Directional signals need a setup that clears the reward to risk floor."""
        if signal.signal not in DIRECTIONAL_SIGNALS:
            return []

        setup = response.get("setup") or {}
        reward_risk = setup.get("rr")
        if reward_risk is None:
            reward_risk = _reward_risk_from_prices(signal, setup)
        if reward_risk is None:
            return ["directional signal requires setup.rr"]
        if reward_risk < self._min_reward_risk:
            return [f"reward to risk {reward_risk:.2f} below minimum {self._min_reward_risk}"]
        return []

    def _hallucination_errors(
        self, response: dict[str, Any], known_patterns: tuple[str, ...]
    ) -> list[str]:
        """Reject reasoning that cites patterns absent from the payload."""
        if not known_patterns:
            return []
        reasoning = str(response.get("reasoning", "")).lower()
        invented = sorted(_cited_patterns(reasoning) - set(known_patterns))
        if invented:
            return [f"reasoning cites unknown patterns: {', '.join(invented)}"]
        return []

    def _soft_checks(self, signal: TradingSignal, response: dict[str, Any]) -> list[str]:
        """Problems that do not reject the response."""
        warnings: list[str] = []
        if signal.signal in DIRECTIONAL_SIGNALS and not response.get("invalidation"):
            warnings.append("directional signal without an invalidation level")
        if signal.signal == Signal.WAIT.value and not response.get("wait_for"):
            warnings.append("WAIT signal without a wait_for condition")
        return warnings


def _setup_errors(setup: Any) -> list[str]:
    """Validate the optional setup block."""
    if not isinstance(setup, dict):
        return ["setup must be an object"]
    errors = [
        f"setup.{name} must be a number"
        for name in ("entry", "stop", "target")
        if name in setup and not _is_number(setup[name])
    ]
    if errors:
        return errors
    entry, stop = setup.get("entry"), setup.get("stop")
    if _is_number(entry) and _is_number(stop) and entry == stop:
        errors.append("setup entry and stop must differ")
    return errors


def _signal_of(response: dict[str, Any]) -> TradingSignal:
    """Build a :class:`TradingSignal` from a validated response."""
    setup = response.get("setup")
    levels = response.get("key_levels") or {}
    return TradingSignal(
        bias=str(response["bias"]),
        confidence=float(response["confidence"]),
        signal=str(response["signal"]),
        primary_evidence=str(response["primary_evidence"]),
        reasoning=str(response["reasoning"]),
        key_levels={
            "support": [float(value) for value in levels.get("support", [])],
            "resistance": [float(value) for value in levels.get("resistance", [])],
        },
        setup={key: float(value) for key, value in setup.items()} if isinstance(setup, dict) else None,
        wait_for=response.get("wait_for"),
        invalidation=str(response.get("invalidation", "")),
        risk_notes=str(response.get("risk_notes", "")),
    )


def _reward_risk_from_prices(signal: TradingSignal, setup: dict[str, Any]) -> float | None:
    """Derive reward to risk from entry, stop and target when ``rr`` is absent."""
    entry, stop, target = setup.get("entry"), setup.get("stop"), setup.get("target")
    if not all(_is_number(value) for value in (entry, stop, target)):
        return None
    risk = abs(float(entry) - float(stop))
    reward = float(target) - float(entry) if signal.signal == Signal.BUY.value else float(entry) - float(target)
    if risk <= 0:
        return None
    return round(reward / risk, 4)


def _cited_patterns(text: str) -> set[str]:
    """Pattern names mentioned in ``text``."""
    return {
        detector.name
        for detector in default_detectors()
        if detector.name in text or detector.label.lower() in text
    }


def _is_number(value: Any) -> bool:
    """``True`` for real numbers, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)