"""Feed interfaces.

A feed is anything that can produce an unbounded stream of domain objects over
time. Three implementations exist (websocket, REST polling, file replay) plus a
synthetic generator, and the pipeline only ever sees these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from ...core.models import BookSnapshot, DepthUpdate, Trade

__all__ = ["DepthFeed", "Feed", "TradeFeed"]

T = TypeVar("T")


class Feed(Generic[T], ABC):
    """An asynchronous stream of :class:`T` values."""

    name: str = "feed"

    @abstractmethod
    def stream(self) -> AsyncIterator[T]:
        """Yield items until the feed is exhausted or the consumer stops."""

    async def aclose(self) -> None:
        """Release resources held by the feed. Overridden when needed."""

    def __aiter__(self) -> AsyncIterator[T]:
        return self.stream()


class TradeFeed(Feed[Trade], ABC):
    """A feed of executed trades."""

    name = "trade-feed"

    @abstractmethod
    def stream(self) -> AsyncIterator[Trade]:
        """Yield normalised trades."""

    async def aclose(self) -> None:
        """Release resources. Trade feeds hold none by default."""

    def snapshot(self) -> BookSnapshot | None:
        """Return the latest order book when this feed also carries depth."""
        return None


class DepthFeed(Feed[DepthUpdate], ABC):
    """A feed of order book updates backed by a local book state."""

    name = "depth-feed"

    @abstractmethod
    def stream(self) -> AsyncIterator[DepthUpdate]:
        """Yield normalised depth updates."""

    @abstractmethod
    def snapshot(self) -> BookSnapshot:
        """Return the book as reconstructed from the updates seen so far."""