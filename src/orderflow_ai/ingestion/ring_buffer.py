"""Fixed capacity ring buffer used to retain recent ticks.

The engine never wants unbounded memory: at tick frequency a single trading
session produces millions of events. A ring buffer keeps the newest ``capacity``
items and silently overwrites the oldest ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

__all__ = ["RingBuffer"]

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Circular buffer with O(1) append and bounded memory use."""

    __slots__ = ("_items", "_capacity", "_size", "_cursor")

    def __init__(self, capacity: int) -> None:
        """Create a buffer that retains at most ``capacity`` items.

        Args:
            capacity: Maximum number of retained items. Must be positive.

        Raises:
            ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self._items: list[T | None] = [None] * capacity
        self._capacity = capacity
        self._size = 0
        self._cursor = 0

    @property
    def capacity(self) -> int:
        """Maximum number of items the buffer can hold."""
        return self._capacity

    def append(self, item: T) -> None:
        """Add ``item``, overwriting the oldest entry when full."""
        self._items[self._cursor] = item
        self._cursor = (self._cursor + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def extend(self, items: Iterable[T]) -> None:
        """Append every item of ``items`` in order."""
        for item in items:
            self.append(item)

    def latest(self, count: int = 1) -> list[T]:
        """Return the ``count`` most recent items, oldest first."""
        count = max(0, min(count, self._size))
        if count == 0:
            return []
        return self[-count:]  # type: ignore[return-value]

    def to_list(self) -> list[T]:
        """Return every retained item, oldest first."""
        return list(self)

    def clear(self) -> None:
        """Drop every retained item."""
        self._items = [None] * self._capacity
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        start = (self._cursor - self._size) % self._capacity
        for offset in range(self._size):
            item = self._items[(start + offset) % self._capacity]
            if item is not None:
                yield item

    def __getitem__(self, index: int) -> T:
        if isinstance(index, slice):  # pragma: no cover - slices not used internally
            return self.to_list()[index]  # type: ignore[return-value]
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError("ring buffer index out of range")
        item = self._items[(self._cursor - self._size + index) % self._capacity]
        if item is None:  # pragma: no cover - defensive
            raise IndexError("ring buffer index out of range")
        return item

    def __bool__(self) -> bool:
        return self._size > 0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"RingBuffer(size={self._size}, capacity={self._capacity})"