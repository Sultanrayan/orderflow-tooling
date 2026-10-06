"""Cross source timestamp alignment.

Trades, depth updates and REST candles arrive with different clocks and
occasionally with out of order timestamps. :class:`TimestampSync` converts every
source onto one epoch-second timeline, clamps regressions and reports the
observed skew so ingestion problems are visible instead of silently corrupting
the analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

__all__ = ["SourceClock", "TimestampSync"]


@dataclass(slots=True)
class SourceClock:
    """Latest observed state of one data source."""

    name: str
    last_timestamp: float = 0.0
    max_seen: float = 0.0
    regressions: int = 0
    dropped: float = 0.0


class TimestampSync:
    """Aligns timestamps coming from several concurrent feeds."""

    __slots__ = ("_clocks", "_reference", "_max_regression_ms", "_max_skew_seconds")

    def __init__(self, max_regression_ms: float = 250.0, max_skew_seconds: float = 2.0) -> None:
        """Create a synchroniser.

        Args:
            max_regression_ms: Largest tolerated backwards step per source. A
                larger regression is clamped back onto the timeline.
            max_skew_seconds: Skew beyond this value is flagged as a warning in
                :meth:`report`.
        """
        self._clocks: dict[str, SourceClock] = {}
        self._reference: float = 0.0
        self._max_regression_ms = max_regression_ms
        self._max_skew_seconds = max_skew_seconds

    def register(self, source: str) -> SourceClock:
        """Register ``source`` and return its clock state."""
        return self._clocks.setdefault(source, SourceClock(name=source))

    def normalize(self, source: str, timestamp: float) -> float:
        """Convert ``timestamp`` into a monotonic epoch-second value.

        Args:
            source: Name of the feed the timestamp came from.
            timestamp: Raw epoch-seconds value. Values above ``1e11`` are
                treated as milliseconds.

        Returns:
            The normalised timestamp. Values that would move the timeline
            backwards are clamped to the last accepted value of that source.
        """
        clock = self.register(source)
        if timestamp > 1e11:
            timestamp /= 1000.0

        clock.max_seen = max(clock.max_seen, timestamp)
        if timestamp < clock.last_timestamp - (self._max_regression_ms / 1000.0):
            clock.regressions += 1
            clock.dropped += clock.last_timestamp - timestamp
            return clock.last_timestamp

        clock.last_timestamp = timestamp
        self._reference = max(self._reference, timestamp)
        return timestamp

    def now(self, source: str = "local") -> float:
        """Return the local wall clock, normalised onto the shared timeline."""
        return self.normalize(source, time.time())

    def synchronize(self, samples: dict[str, float]) -> float:
        """Align a batch of timestamps and return the newest normalised value.

        Every sample is passed through :meth:`normalize`; the result is the
        maximum of the normalised values, which keeps downstream ordering
        deterministic even when feeds disagree slightly.
        """
        normalised = [self.normalize(name, value) for name, value in samples.items()]
        return max(normalised, default=self._reference)

    def lag_seconds(self, source: str) -> float:
        """Difference between the newest value of ``source`` and the timeline."""
        clock = self._clocks.get(source)
        if clock is None:
            return 0.0
        return max(0.0, self._reference - clock.last_timestamp)

    def report(self) -> dict[str, dict[str, float | int]]:
        """Per source diagnostics: lag, regressions and skipped drift."""
        return {
            name: {
                "last_timestamp": clock.last_timestamp,
                "lag_seconds": round(self.lag_seconds(name), 6),
                "regressions": clock.regressions,
                "dropped_seconds": round(clock.dropped, 6),
                "stale": self.lag_seconds(name) > self._max_skew_seconds,
            }
            for name, clock in sorted(self._clocks.items())
        }