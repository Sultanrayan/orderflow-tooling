"""Prompt assembly.

The prompt has two parts. The static block holds the role, the trading rules and
the legend that explains the array positions; it never changes between calls,
which is what makes provider side prompt caching effective. The dynamic block is
the optimised payload for one bar.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..core.analysis import Analysis
from .formatter import PayloadBuilder
from .optimizer import OptimizationReport, estimate_tokens

__all__ = ["PromptBuilder", "PromptMessage", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You are an order flow analyst. Protect capital first. "
    "NO_TRADE and WAIT are valid outcomes and are preferred over a weak setup. "
    "Only report patterns that appear in the payload; never invent one. "
    "Answer with a single JSON object and no other text."
)

RESPONSE_CONTRACT = (
    '{"bias":"bullish|bearish|neutral","confidence":0-100,"signal":"BUY|SELL|NO_TRADE|WAIT",'
    '"primary_evidence":"patterns|primitives|context","reasoning":"one or two sentences",'
    '"key_levels":{"support":[numbers],"resistance":[numbers]},'
    '"setup":{"entry":number,"stop":number,"target":number,"rr":number},'
    '"wait_for":string|null,"invalidation":string,"risk_notes":string}'
)

LEGEND = (
    "m=[symbol,timeframe,bar_open,price,tier] "
    "d=[delta,delta_ratio,dir,cvd,cvd_slope,momentum,divergence] "
    "pat=[[name,price,direction,confidence,bars_ago]] "
    "lvl=[support[],resistance[],poc] "
    "c=[structure,bos,choch,session,volatility_regime,mtf_score] "
    "s=[[open,high,low,close],volume,delta,levels[[price,bid,ask]],"
    "profile[poc,vah,val,hvn,lvn],migration[dir,strength]] "
    "p.ob=[bid_depth,ask_depth,ratio,dir] p.top=[price,side,ratio] "
    "q=[[intensity_ratio,percentile,level],[absorption_detected,confidence,side]] "
    "fb=[score,tier,[[component,points]]] "
    "direction values: bullish|bearish|neutral. side values: buy|sell. "
    "Empty strings, zero and missing keys mean \"not measured\", never zero risk."
)


@dataclass(frozen=True, slots=True)
class PromptMessage:
    """One chat message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Provider independent representation."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class PromptBuild:
    """Result of building a prompt."""

    messages: tuple[PromptMessage, ...]
    payload: dict
    tier: str
    report: OptimizationReport
    static_hash: str

    @property
    def token_estimate(self) -> int:
        """Estimated tokens for the whole prompt."""
        return sum(estimate_tokens(message.content) for message in self.messages)


class PromptBuilder:
    """Builds chat messages for a provider request."""

    def __init__(
        self,
        payload_builder: PayloadBuilder | None = None,
        include_legend: bool = True,
    ) -> None:
        """Create the builder.

        Args:
            payload_builder: Payload source. A default one is created when
                omitted.
            include_legend: Append the array legend to the static block.
        """
        self._payloads = payload_builder or PayloadBuilder()
        self._include_legend = include_legend
        self._static_cache: dict[str, str] = {}

    @property
    def payload_builder(self) -> PayloadBuilder:
        """Payload builder in use."""
        return self._payloads

    def static_block(self) -> str:
        """The stable part of the prompt, built once and cached in-process."""
        cached = self._static_cache.get("static")
        if cached is None:
            cached = self._compose_static_block()
            self._static_cache["static"] = cached
        return cached

    def static_hash(self) -> str:
        """SHA-256 prefix of the static block, usable as a prompt cache key."""
        digest = hashlib.sha256(self.static_block().encode("utf-8")).hexdigest()
        return digest[:16]

    def build(self, analysis: Analysis, tier: str | None = None) -> PromptBuild:
        """Build the messages for ``analysis``.

        Args:
            analysis: The analysis to describe.
            tier: Force a payload tier.

        Returns:
            The messages plus the payload, report, tier and static cache key.
        """
        build = self._payloads.build_optimized(analysis, tier)
        messages = (
            PromptMessage("system", self.static_block()),
            PromptMessage("user", self._payloads.optimizer.serialize(build.payload)),
        )
        return PromptBuild(
            messages=messages,
            payload=build.payload,
            tier=build.tier,
            report=build.report,
            static_hash=self.static_hash(),
        )

    def _compose_static_block(self) -> str:
        """Assemble the system prompt from its stable parts."""
        parts = [SYSTEM_PROMPT, f"Answer schema: {RESPONSE_CONTRACT}"]
        if self._include_legend:
            parts.append(f"Payload legend: {LEGEND}")
        return "\n".join(parts)