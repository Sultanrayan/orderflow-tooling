"""Trading session context.

Sessions are defined in UTC, which is the only timezone the exchange feeds
report. Times of day follow the usual FX conventions: London opens at 07:00 UTC
and New York at 12:30 UTC, so their overlap is 12:30 to 16:00.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum

from .models import SessionContext

__all__ = ["Session", "session_of", "session_window"]


class Session(str, Enum):
    """Named trading sessions."""

    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"

    @property
    def is_major(self) -> bool:
        """``True`` for the two high liquidity western sessions."""
        return self in {Session.LONDON, Session.NEW_YORK}


# (start_hour, end_hour) in UTC, half open: start is inclusive, end exclusive.
WINDOWS: dict[Session, tuple[int, int]] = {
    Session.ASIA: (0, 7),
    Session.LONDON: (7, 16),
    Session.NEW_YORK: (12, 21),
}

OPENING_HOURS = 30  # minutes after a session open treated as its opening window


def session_of(moment: dt.datetime) -> Session:
    """Return the primary session for ``moment``."""
    hour = moment.hour
    if 7 <= hour < 16:
        return Session.LONDON
    if 12 <= hour < 21:
        return Session.NEW_YORK
    if 0 <= hour < 7:
        return Session.ASIA
    return Session.OFF_HOURS


def active_sessions(moment: dt.datetime) -> tuple[Session, ...]:
    """Return every session open at ``moment``."""
    hour = moment.hour
    return tuple(
        session for session, (start, end) in WINDOWS.items() if start <= hour < end
    )


def session_window(moment: dt.datetime) -> float:
    """Minutes elapsed since the current session opened."""
    session = session_of(moment)
    if session is Session.OFF_HOURS:
        return 0.0
    start_hour = WINDOWS[session][0]
    return (moment.hour - start_hour) * 60 + moment.minute


def build_session_context(moment: dt.datetime) -> SessionContext:
    """Build the session context for ``moment``."""
    primary = session_of(moment)
    active = active_sessions(moment)
    return SessionContext(
        primary=primary.value,
        active=tuple(session.value for session in active),
        is_london_open=_is_opening(moment, Session.LONDON),
        is_new_york_open=_is_opening(moment, Session.NEW_YORK),
        minutes_into_session=session_window(moment),
    )


def _is_opening(moment: dt.datetime, session: Session) -> bool:
    """``True`` within the first :data:`OPENING_HOURS` of ``session``."""
    start_hour = WINDOWS[session][0]
    elapsed = moment.hour * 60 + moment.minute - start_hour * 60
    return start_hour <= moment.hour < start_hour + 2 and 0 <= elapsed < OPENING_HOURS