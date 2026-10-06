"""Replay based backtesting.

The backtester re-runs the pipeline over recorded ticks, turns every signal into
a paper trade against the next bar, and reports performance. It never assumes a
win: costs, slippage and the "no trade" outcome are all modelled explicitly.
"""

from __future__ import annotations

from .engine import BacktestEngine, BacktestResult, SignalRule
from .metrics import PerformanceMetrics
from .signals import PaperTrader, SignalAction, Trade

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PaperTrader",
    "PerformanceMetrics",
    "SignalAction",
    "SignalRule",
    "Trade",
]