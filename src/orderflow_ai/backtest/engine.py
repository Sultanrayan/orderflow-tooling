"""Backtest engine.

The engine replays recorded data through the very same pipeline used in live
mode, so a backtest measures the real decision path: primitives, patterns,
context and the signal derived from them. No hidden shortcuts.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from ..core.analysis import Analysis
from ..core.config import OrderFlowConfig
from ..core.models import Candle
from ..core.pipeline import OrderFlowPipeline
from ..output.schema import TradingSignal
from .metrics import PerformanceMetrics
from .signals import PaperTrader, SignalAction, Trade

__all__ = ["BacktestEngine", "BacktestResult", "SignalInput", "SignalRule"]

logger = logging.getLogger(__name__)

SignalInput = tuple[Analysis, TradingSignal | None]


@dataclass(frozen=True, slots=True)
class SignalRule:
    """Maps an analysis to a paper trading action.

    The default rule only trades high confidence patterns, which is the
    behaviour the specification recommends for small positions.
    """

    min_confidence: float = 75.0
    allowed_patterns: tuple[str, ...] = ()

    def accepts(self, analysis: Analysis) -> bool:
        """``True`` when the analysis passes every gate of the rule."""
        if not analysis.has_pattern or analysis.confidence < self.min_confidence:
            return False
        if self.allowed_patterns:
            return any(name in self.allowed_patterns for name in analysis.pattern_names)
        return True

    def action_for(self, signal: TradingSignal | None) -> SignalAction:
        """Return the action for a validated signal."""
        if signal is None:
            return SignalAction.FLAT
        return SignalAction.parse(signal.signal)


@dataclass(slots=True)
class BacktestResult:
    """Outcome of one backtest run."""

    metrics: PerformanceMetrics
    trades: tuple[Trade, ...] = ()
    bars: int = 0
    signals: int = 0
    rejected_signals: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """One line summary suitable for a log."""
        return (
            f"bars={self.bars} signals={self.signals} trades={self.metrics.trades} "
            f"win_rate={self.metrics.win_rate}% pnl={self.metrics.total_pnl} "
            f"profit_factor={self.metrics.profit_factor} max_drawdown={self.metrics.max_drawdown}"
        )


class BacktestEngine:
    """Runs a pipeline over recorded data and paper trades the result."""

    def __init__(
        self,
        config: str | Path | OrderFlowConfig | None = None,
        rule: SignalRule | None = None,
        capital: float = 10_000.0,
        risk_fraction: float = 0.01,
        fee_bps: float = 4.0,
        slippage_bps: float = 1.0,
    ) -> None:
        """Create the engine.

        Args:
            config: Pipeline configuration. ``pipeline.mode`` should be ``replay``
                or ``synthetic``.
            rule: Signal filter. Defaults to :class:`SignalRule`.
            capital: Starting notional.
            risk_fraction: Fraction of capital risked per trade.
            fee_bps: Round trip fee in basis points.
            slippage_bps: Entry and exit slippage in basis points.
        """
        self.config = OrderFlowConfig.load(config)
        self.rule = rule or SignalRule()
        self.capital = capital
        self.risk_fraction = risk_fraction
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps

    async def run(
        self,
        signals: AsyncIterator[SignalInput] | None = None,
        max_bars: int | None = None,
    ) -> BacktestResult:
        """Replay the data source and paper trade every accepted signal.

        Args:
            signals: Pre-computed ``(analysis, signal)`` pairs. When omitted the
                engine derives a neutral setup from the analysis itself, which
                keeps backtests runnable without an AI provider.
            max_bars: Stop after this many bars.

        Returns:
            The result, including realised trades and metrics.
        """
        trader = PaperTrader(
            capital=self.capital,
            risk_fraction=self.risk_fraction,
            fee_bps=self.fee_bps,
            slippage_bps=self.slippage_bps,
        )
        counts = _RunCounts()
        last_candle: Candle | None = None

        stream = signals if signals is not None else self._baseline_signals()
        async for analysis, signal in stream:
            counts.bars += 1
            grade = _grade(analysis)
            counts.tiers[grade] = counts.tiers.get(grade, 0) + 1
            candle = _candle_of(analysis)
            last_candle = candle

            action = SignalAction.FLAT
            if self.rule.accepts(analysis):
                action = self.rule.action_for(signal)
            else:
                counts.rejected += 1

            if action is not SignalAction.FLAT:
                counts.traded += 1
                setup = (signal.setup if signal else None) or {}
                trader.on_signal(
                    action,
                    candle,
                    entry=setup.get("entry"),
                    stop=setup.get("stop"),
                    target=setup.get("target"),
                )
            else:
                trader.on_signal(SignalAction.FLAT, candle)

            if max_bars is not None and counts.bars >= max_bars:
                break

        if last_candle is not None:
            trader.close_open(last_candle)

        return BacktestResult(
            metrics=PerformanceMetrics.from_trades(trader.trades, self.capital),
            trades=tuple(trader.trades),
            bars=counts.bars,
            signals=counts.traded,
            rejected_signals=counts.rejected,
            tier_counts=dict(counts.tiers),
        )

    async def _baseline_signals(self) -> AsyncIterator[SignalInput]:
        """Yield analyses paired with the neutral setup they imply.

        Without a model there is no trade plan, so entry, stop and target are
        derived from the average bar range. This measures pattern detection, not
        a discretionary strategy, which is exactly what a backtest of this
        engine should isolate.
        """
        pipeline = OrderFlowPipeline(self.config)
        async for analysis in pipeline.stream():
            yield analysis, baseline_setup(analysis)


@dataclass(slots=True)
class _RunCounts:
    """Mutable counters used while streaming bars."""

    bars: int = 0
    traded: int = 0
    rejected: int = 0
    tiers: dict[str, int] = field(default_factory=dict)


def baseline_setup(analysis: Analysis) -> TradingSignal | None:
    """Build a neutral setup from an analysis, or ``None`` without a pattern."""
    if not analysis.has_pattern:
        return None
    bullish = analysis.direction == "bullish"
    entry = analysis.price
    span = max(analysis.context.volatility.average_range, entry * 0.0005, 1e-9)
    stop = entry - span if bullish else entry + span
    target = entry + span * 2 if bullish else entry - span * 2
    return TradingSignal(
        bias=analysis.direction,
        confidence=analysis.confidence,
        signal="BUY" if bullish else "SELL",
        primary_evidence="patterns",
        reasoning="baseline setup derived from the average bar range",
        key_levels={
            "support": list(analysis.context.levels.support),
            "resistance": list(analysis.context.levels.resistance),
        },
        setup={
            "entry": entry,
            "stop": stop,
            "target": target,
            "rr": abs(target - entry) / abs(entry - stop),
        },
        invalidation=f"close beyond {stop:.6g}",
    )


def _candle_of(analysis: Analysis) -> Candle:
    """Project the analysed bar onto a candle for the paper trader."""
    if analysis.bars:
        return analysis.bars[-1].to_candle()
    return Candle(
        open_time=analysis.open_time,
        open=analysis.price,
        high=analysis.price,
        low=analysis.price,
        close=analysis.price,
        volume=0.0,
        close_time=analysis.open_time,
    )


def _grade(analysis: Analysis) -> str:
    """Confidence grade used for the run histogram."""
    if not analysis.has_pattern:
        return "no_pattern"
    if analysis.confidence >= 75:
        return "high"
    if analysis.confidence >= 50:
        return "medium"
    return "low"