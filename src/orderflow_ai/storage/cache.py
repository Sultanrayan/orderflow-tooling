"""Optional infrastructure integrations.

Only Redis is wired up, and only for caching: the engine never requires a
running Redis instance. :class:`Cache` degrades to an in-process dictionary so
local runs and tests behave identically to production, minus persistence.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

__all__ = ["Cache", "MemoryCache", "RedisCache", "build_cache"]

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300
ENV_REDIS_URL = "REDIS_URL"


class Cache:
    """Interface for a key/value cache with a time to live."""

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key``, or ``None``."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Store ``value`` under ``key``."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Remove ``key`` if present."""
        raise NotImplementedError

    def ping(self) -> bool:
        """Return ``True`` when the backend is reachable."""
        raise NotImplementedError


@dataclass(slots=True)
class _Entry:
    value: Any
    expires_at: float


class MemoryCache(Cache):
    """In-process cache with expiry, used as the default backend."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def get(self, key: str) -> Any | None:
        """Return the cached value, dropping it when expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at and entry.expires_at < time.time():
            del self._entries[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Store ``value`` with an optional time to live."""
        expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else 0.0
        self._entries[key] = _Entry(value=value, expires_at=expires_at)

    def delete(self, key: str) -> None:
        """Remove ``key``."""
        self._entries.pop(key, None)

    def ping(self) -> bool:
        """The in-process cache is always available."""
        return True


class RedisCache(Cache):
    """Redis backed cache. Falls back to memory when Redis is unreachable."""

    def __init__(self, url: str, client: Any | None = None, prefix: str = "orderflow") -> None:
        """Create the cache.

        Args:
            url: Redis connection URL.
            client: Pre-built client, mainly for tests.
            prefix: Key prefix, keeping the engine's keys namespaced.
        """
        self.url = url
        self._prefix = prefix
        self._client = client or self._connect(url)

    @property
    def available(self) -> bool:
        """``True`` when a real Redis client is in use."""
        return self._client is not None

    def get(self, key: str) -> Any | None:
        """Return the cached JSON value for ``key``."""
        if self._client is None:
            return None
        try:
            raw = self._client.get(self._key(key))
        except Exception as exc:  # noqa: BLE001 - caching must never break analysis
            logger.warning("redis get failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    def set(self, key: str, value: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        """Store ``value`` as JSON with a time to live."""
        if self._client is None:
            return
        try:
            self._client.setex(self._key(key), ttl_seconds, json.dumps(value, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis set failed: %s", exc)

    def delete(self, key: str) -> None:
        """Remove ``key``."""
        if self._client is None:
            return
        try:
            self._client.delete(self._key(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis delete failed: %s", exc)

    def ping(self) -> bool:
        """Return ``True`` when Redis answers."""
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception as exc:  # noqa: BLE001
            logger.debug("redis ping failed: %s", exc)
            return False

    def _key(self, key: str) -> str:
        """Namespace a key."""
        return f"{self._prefix}:{key}"

    @staticmethod
    def _connect(url: str) -> Any | None:
        """Connect to Redis, returning ``None`` when it is not installed or up."""
        try:
            import redis
        except ImportError:
            logger.info("redis package not installed; caching disabled")
            return None
        try:
            return redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not connect to redis at %s: %s", url, exc)
            return None


def build_cache(url: str | None = None) -> Cache:
    """Return a Redis cache when a URL is configured, otherwise memory."""
    target = url or os.environ.get(ENV_REDIS_URL)
    if not target:
        return MemoryCache()
    cache = RedisCache(target)
    return cache if cache.ping() else MemoryCache()