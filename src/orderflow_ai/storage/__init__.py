"""Optional infrastructure: caching backed by Redis when it is available."""

from __future__ import annotations

from .cache import Cache, MemoryCache, RedisCache, build_cache

__all__ = ["Cache", "MemoryCache", "RedisCache", "build_cache"]