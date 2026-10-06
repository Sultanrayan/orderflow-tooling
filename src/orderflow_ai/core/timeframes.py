"""Supported candle timeframes and the time arithmetic built on top of them."""

from __future__ import annotations

import datetime as dt
from enum import Enum

__all__ = ["Timeframe", "to_epoch_seconds", "floor_timestamp", "bar_open_time"]


class Timeframe(str, Enum):
    """Candle durations supported by the engine.

    The enum derives from :class:`str` so a timeframe can be used anywhere a
    plain string is expected (configuration files, JSON payloads, logs).
    """

    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        """Length of one bar in seconds."""
        return _SECONDS[self]

    @property
    def milliseconds(self) -> int:
        """Length of one bar in milliseconds."""
        return self.seconds * 1000

    @property
    def minutes(self) -> float:
        """Length of one bar in minutes, useful for annualisation maths."""
        return self.seconds / 60

    @classmethod
    def from_string(cls, value: Timeframe | str) -> Timeframe:
        """Parse ``value`` into a :class:`Timeframe`.

        Args:
            value: A timeframe member or a label such as ``"1m"``, ``"4h"``.

        Returns:
            The matching timeframe.

        Raises:
            ValueError: If the label is not a supported timeframe.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            supported = ", ".join(sorted(member.value for member in cls))
            raise ValueError(f"Unsupported timeframe {value!r}. Supported: {supported}") from exc

    @classmethod
    def ranked(cls) -> list[Timeframe]:
        """All timeframes ordered from shortest to longest."""
        return sorted(cls, key=lambda timeframe: timeframe.seconds)

    def higher_than(self, other: Timeframe) -> bool:
        """Return ``True`` when this timeframe is strictly longer than ``other``."""
        return self.seconds > other.seconds

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)


_SECONDS: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M3: 180,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3600,
    Timeframe.H2: 7200,
    Timeframe.H4: 14_400,
    Timeframe.H6: 21_600,
    Timeframe.H8: 28_800,
    Timeframe.H12: 43_200,
    Timeframe.D1: 86_400,
    Timeframe.W1: 604_800,
}


def to_epoch_seconds(value: dt.datetime | float | int) -> float:
    """Normalise a timestamp to epoch seconds.

    Naive datetimes are interpreted as UTC, which matches every exchange feed
    used by this engine.

    Args:
        value: A timezone-aware datetime, or an epoch value in seconds or
            milliseconds. Values above ``1e11`` are treated as milliseconds.

    Returns:
        The timestamp in epoch seconds.
    """
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.timestamp()
    numeric = float(value)
    return numeric / 1000.0 if abs(numeric) > 1e11 else numeric


def floor_timestamp(epoch_seconds: float, timeframe: Timeframe) -> float:
    """Return the opening timestamp of the bar that contains ``epoch_seconds``."""
    return epoch_seconds - (epoch_seconds % timeframe.seconds)


def bar_open_time(epoch_seconds: float, timeframe: Timeframe) -> dt.datetime:
    """Return the UTC open time of the bar containing ``epoch_seconds``."""
    return dt.datetime.fromtimestamp(
        floor_timestamp(epoch_seconds, timeframe), tz=dt.UTC
    )