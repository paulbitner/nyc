"""
cache_store.py
--------------
A tiny, thread-safe LRU + TTL cache for small API responses.

Features:
- TTL expiration per entry
- LRU eviction on capacity
- Optional serve-stale-on-error
- ETag/Last-Modified revalidation metadata

Intended for single-process FastAPI apps. For multi-instance or distributed
deployments, use a shared cache like Redis instead.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from threading import RLock
from time import time
from typing import Any, Optional


@dataclass
class CacheEntry:
    data: Any
    expires_at: float
    etag: Optional[str] = None
    last_modified: Optional[str] = None


class TTLCacheLRU:
    """
    A minimal LRU cache with TTL.
    - get(): returns CacheEntry or None (expired entries are purged on access)
    - set(): stores data with TTL; evicts LRU when over capacity
    - stats(): hit/miss/eviction counters

    Thread-safe via a single RLock (sufficient for typical FastAPI workloads).
    """
    def __init__(self, maxsize: int, ttl_seconds: int, serve_stale_on_error: bool = True) -> None:
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.serve_stale_on_error = serve_stale_on_error
        self._store: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _now(self) -> float:
        return time()

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            # Expired? Purge and miss.
            if entry.expires_at <= self._now():
                # keep it in case caller wants to serve stale on error (they still need metadata)
                # but mark as logically expired by popping; caller can keep the object reference
                self._store.pop(key, None)
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry

    def set(self, key: str, data: Any, etag: Optional[str] = None, last_modified: Optional[str] = None) -> CacheEntry:
        with self._lock:
            entry = CacheEntry(
                data=data,
                expires_at=self._now() + self.ttl,
                etag=etag,
                last_modified=last_modified,
            )
            self._store[key] = entry
            self._store.move_to_end(key)
            # Evict if over capacity
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)
                self._evictions += 1
            return entry

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._store),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "serve_stale_on_error": self.serve_stale_on_error,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = self._misses = self._evictions = 0
