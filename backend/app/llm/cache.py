"""Deterministic-response cache for LLM calls.

The MVP rule is "stop making multiple unnecessary provider calls": the same
(system prompt, user message, temperature) always answers the same question, and
the demo replays the same handful of questions constantly (page reloads, the
incident replay's three candidate queries, the seed story). Every one of those
is a cache hit here — zero provider requests, zero quota consumed, and the
latency/token counters still report the *original* call honestly.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple

DEFAULT_TTL_SECONDS = 900
DEFAULT_MAX_ENTRIES = 512


def make_key(*parts: str) -> str:
    """Stable cache key for a request, independent of process/restart."""
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class TTLCache:
    """Insertion-ordered LRU with per-entry TTL and hit/miss counters.

    Not thread-safe on its own, but FastAPI runs one event loop and every
    mutation below is a plain dict operation with no awaits in between.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._entries: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Tuple[bool, Any]:
        if self.ttl_seconds <= 0:
            self.misses += 1
            return False, None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return False, None
        stored_at, value = entry
        if time.monotonic() - stored_at > self.ttl_seconds:
            del self._entries[key]
            self.misses += 1
            return False, None
        self._entries.move_to_end(key)
        self.hits += 1
        return True, value

    def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return
        self._entries[key] = (time.monotonic(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "entries": len(self._entries),
            "ttl_seconds": self.ttl_seconds,
            "max_entries": self.max_entries,
        }

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0
