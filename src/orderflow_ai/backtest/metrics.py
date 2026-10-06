"""Performance metrics for a trade list.

Every metric is computed from realised trades only, so the numbers stay
reproducible and easy to audit. ``to_dict`` returns JSON friendly output for
reports and API responses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .signals import Trade

__all__ = ["PerformanceMetrics"]


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Summary statistics of a backtest."""

    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy: float
    total_pnl: float
    average_r: float
    best_trade: float
    worst_trade: float
    max_drawdown: float
    average_bars_held: float
    fees_paid: float

    @classmethod
    def from_trades(cls, trades: Sequence[Trade], starting_equity: float = 0.0) -> PerformanceMetrics:
        """Compute the metrics for ``trades``.

        Args:
            trades: Closed trades in chronological order.
            starting_equity: Equity before the first trade, used for drawdown.

        Returns:
            The metrics. All values are zero when no trade was taken.
        """
        if not trades:
            return cls.empty()

        wins = [trade for trade in trades if trade.is_win]
        losses = [trade for trade in trades if not trade.is_win]
        gross_profit = sum(trade.pnl for trade in wins)
        gross_loss = abs(sum(trade.pnl for trade in losses))

        return cls(
            trades=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=round(len(wins) / len(trades) * 100, 2),
            profit_factor=round(gross_profit / gross_loss, 3) if gross_loss else 0.0,
            expectancy=round(sum(trade.pnl for trade in trades) / len(trades), 8),
            total_pnl=round(sum(trade.pnl for trade in trades), 8),
            average_r=round(sum(trade.r_multiple for trade in trades) / len(trades), 3),
            best_trade=round(max(trade.pnl for trade in trades), 8),
            worst_trade=round(min(trade.pnl for trade in trades), 8),
            max_drawdown=_max_drawdown(trades, starting_equity),
            average_bars_held=round(sum(trade.bars_held for trade in trades) / len(trades), 2),
            fees_paid=round(sum(trade.fees for trade in trades), 8),
        )

    @classmethod
    def empty(cls) -> PerformanceMetrics:
        """Metrics for a run without any trade."""
        return cls(
            trades=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_pnl=0.0,
            average_r=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            max_drawdown=0.0,
            average_bars_held=0.0,
            fees_paid=0.0,
        )

    def to_dict(self) -> dict[str, float | int]:
        """JSON friendly view of the metrics."""
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "total_pnl": self.total_pnl,
            "average_r": self.average_r,
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
            "max_drawdown": self.max_drawdown,
            "average_bars_held": self.average_bars_held,
            "fees_paid": self.fees_paid,
        }


def _max_drawdown(trades: Sequence[Trade], starting_equity: float) -> float:
    """Largest peak to trough decline of the equity curve."""
    equity = starting_equity
    peak = equity
    drawdown = 0.0
    for trade in trades:
        equity += trade.pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return round(drawdown, 8)