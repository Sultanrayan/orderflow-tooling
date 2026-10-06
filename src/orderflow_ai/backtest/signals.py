"""Paper trading of a signal stream.

Fills are simulated bar by bar with a fixed slippage and fee model. Both the
stop and the target can be touched inside one bar; the pessimistic assumption
(stop first) is used, because bar data cannot prove the order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.models import Candle

__all__ = ["PaperTrader", "SignalAction", "Trade"]


class SignalAction(str, Enum):
    """Actions the paper trader understands."""

    BUY = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"

    @classmethod
    def parse(cls, value: str | SignalAction | None) -> SignalAction:
        """Coerce arbitrary input into a :class:`SignalAction`.

        Unknown values, including ``None``, become :attr:`FLAT`.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError:
            return cls.FLAT


@dataclass(slots=True)
class Trade:
    """One simulated round trip.

    Attributes:
        risk: Price distance between entry and stop.
        risk_amount: Currency put at risk, equal to ``risk * quantity``.
        r_multiple: Profit expressed in units of ``risk_amount``.
    """

    side: str
    entry: float
    stop: float
    target: float
    exit_price: float
    quantity: float
    risk: float
    risk_amount: float
    fees: float
    pnl: float
    bars_held: int
    exit_reason: str
    opened_at: str = ""

    @property
    def is_win(self) -> bool:
        """``True`` when the trade closed with a profit."""
        return self.pnl > 0

    @property
    def r_multiple(self) -> float:
        """Result expressed in units of the initial risk."""
        return self.pnl / self.risk_amount if self.risk_amount > 0 else 0.0


@dataclass(slots=True)
class _OpenPosition:
    """Internal bookkeeping for an open position."""

    side: SignalAction
    entry: float
    stop: float
    target: float
    bars: int = 0
    opened_at: str = ""


@dataclass(slots=True)
class PaperTrader:
    """Applies signals to bars and records the resulting trades.

    Attributes:
        capital: Starting notional, used for position sizing.
        risk_fraction: Fraction of capital risked per trade.
        fee_bps: Round trip fee in basis points of notional.
        slippage_bps: Entry and exit slippage in basis points.
    """

    capital: float = 10_000.0
    risk_fraction: float = 0.01
    fee_bps: float = 4.0
    slippage_bps: float = 1.0
    trades: list[Trade] = field(default_factory=list)
    position: _OpenPosition | None = None

    @property
    def equity(self) -> float:
        """Capital plus realised profit."""
        return self.capital + sum(trade.pnl for trade in self.trades)

    @property
    def has_open_trade(self) -> bool:
        """``True`` while a position is open."""
        return self.position is not None

    def on_signal(
        self,
        action: str | SignalAction,
        candle: Candle,
        entry: float | None = None,
        stop: float | None = None,
        target: float | None = None,
    ) -> Trade | None:
        """Advance the open position, then optionally open a new one.

        Args:
            action: ``BUY``, ``SELL`` or ``FLAT``. Anything else is treated as
                ``FLAT``.
            candle: The bar the signal was produced on.
            entry: Requested entry price. Defaults to the candle close.
            stop: Protective stop. Required to open a position.
            target: Profit target. Optional.

        Returns:
            The trade closed by this call, if any.
        """
        closed = self.on_bar(candle)
        signal = SignalAction.parse(action)
        if signal is SignalAction.FLAT or stop is None or self.has_open_trade:
            return closed

        self.position = _OpenPosition(
            side=signal,
            entry=entry or candle.close,
            stop=stop,
            target=target or 0.0,
            opened_at=candle.open_time.isoformat(timespec="minutes"),
        )
        return closed

    def on_bar(self, candle: Candle) -> Trade | None:
        """Advance an open position by one bar.

        Returns:
            The trade closed by this bar, if any.
        """
        position = self.position
        if position is None:
            return None

        reason, price = _exit_for(position, candle)
        if reason is None:
            position.bars += 1
            return None

        trade = self._close(position, price, reason)
        self.position = None
        return trade

    def close_open(self, candle: Candle, price: float | None = None) -> Trade | None:
        """Force close the open position at ``price`` or the candle close."""
        position = self.position
        if position is None:
            return None
        trade = self._close(position, price or candle.close, "manual")
        self.position = None
        return trade

    def _close(self, position: _OpenPosition, exit_price: float, reason: str) -> Trade:
        """Book a closed trade after slippage and fees."""
        slip = self.slippage_bps / 10_000
        filled_entry = position.entry * (1 + slip)
        filled_exit = exit_price * (1 - slip)
        risk = abs(filled_entry - position.stop)
        quantity = (self.capital * self.risk_fraction) / risk if risk > 0 else 0.0
        direction = 1.0 if position.side is SignalAction.BUY else -1.0
        gross = (filled_exit - filled_entry) * quantity * direction
        fees = (filled_entry + filled_exit) * quantity * self.fee_bps / 10_000

        trade = Trade(
            side=position.side.value,
            entry=round(filled_entry, 8),
            stop=position.stop,
            target=position.target,
            exit_price=round(filled_exit, 8),
            quantity=round(quantity, 8),
            risk=round(risk, 8),
            risk_amount=round(risk * quantity, 8),
            fees=round(fees, 8),
            pnl=round(gross - fees, 8),
            bars_held=position.bars,
            exit_reason=reason,
            opened_at=position.opened_at,
        )
        self.trades.append(trade)
        return trade


def _exit_for(position: _OpenPosition, candle: Candle) -> tuple[str | None, float]:
    """Decide whether a candle triggers an exit, and at which price."""
    if position.side is SignalAction.BUY:
        stop_hit = candle.low <= position.stop
        target_hit = bool(position.target) and candle.high >= position.target
    else:
        stop_hit = candle.high >= position.stop
        target_hit = bool(position.target) and candle.low <= position.target

    # Pessimistic ordering: when both levels sit inside the bar, assume the stop
    # was reached first, because bar data cannot prove the order.
    if stop_hit:
        return "stop", position.stop
    if target_hit:
        return "target", position.target
    return None, 0.0