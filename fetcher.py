# fetcher.py
import os
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

from cache_store import TTLCacheLRU

# -----------------------------------------------------------
# Environment & logging
# -----------------------------------------------------------
load_dotenv()
logger = logging.getLogger("uvicorn.error")

BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
APP_TOKEN = os.getenv("NYC_APP_TOKEN")

# Cache config
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "60"))        # default 60s
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "256"))             # default 256 keys
CACHE_SERVE_STALE_ON_ERROR = os.getenv("CACHE_SERVE_STALE_ON_ERROR", "1") == "1"

# Process / instance identifiers for debugging
PID = os.getpid()

# Singleton cache + upstream-call counter
_cache = TTLCacheLRU(
    maxsize=CACHE_MAX_SIZE,
    ttl_seconds=CACHE_TTL_SECONDS,
    serve_stale_on_error=CACHE_SERVE_STALE_ON_ERROR,
)
UPSTREAM_CALLS = 0  # incremented every time we actually hit Socrata

logger.info(
    "NYC_APP_TOKEN loaded? %s | cache ttl=%ss, max=%s, stale_on_error=%s",
    "yes" if APP_TOKEN else "no",
    CACHE_TTL_SECONDS,
    CACHE_MAX_SIZE,
    CACHE_SERVE_STALE_ON_ERROR,
)
logger.info("CACHE INIT → pid=%s cache_id=%s", PID, id(_cache))


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def _mask_token(token: Optional[str]) -> str:
    if not token:
        return "None"
    return f"{token[:4]}…{token[-4:]}" if len(token) > 8 else "****"


def _cache_key(limit: int, borough: Optional[str]) -> str:
    # Keep the key stable regardless of caller’s casing
    b = (borough or "ALL").upper()
    return f"limit={limit}|borough={b}"


# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------
def fetch_311_data(limit: int = 10, borough: Optional[str] = None):
    """
    Fetch recent NYC 311 service requests with a configurable limit and optional borough filter.
    Uses an in-process TTL+LRU cache. On a cache HIT, no upstream request is performed.

    Args:
        limit: number of rows to fetch (default 10)
        borough: optional borough (case-insensitive; e.g., "MANHATTAN")

    Returns:
        Parsed JSON (list) on success, or {"error": "..."} on failure.
    """
    # Normalize inputs early (ensures stable cache keys & params)
    borough_norm = borough.upper() if borough else None
    key = _cache_key(limit, borough_norm)

    logger.info("FETCH start → pid=%s cache_id=%s", PID, id(_cache))
    logger.info("KEY → limit=%s borough=%s key=%s", limit, borough_norm or "ALL", key)

    # 1) Try cache (fresh only). On HIT, return immediately (NO upstream call).
    entry = _cache.get(key)
    if entry:
        logger.info("CACHE HIT → key=%s", key)
        return entry.data

    logger.info("CACHE MISS → key=%s", key)

    # 2) Build request for upstream
    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    params = {
        "$limit": limit,
        "$order": "created_date DESC",
    }
    if borough_norm:
        params["borough"] = borough_norm

    # 3) Perform upstream call (MISS path only)
    logger.info("UPSTREAM CALL → key=%s (about to request)", key)
    try:
        resp = requests.get(BASE_URL, headers=headers, params=params, timeout=20)
    except requests.RequestException as e:
        # Optional: serve stale if available (requires keeping expired entries separately)
        logger.error("Network error on upstream fetch for %s: %s", key, str(e))
        return {"error": f"Network error: {str(e)}"}

    # Record that we actually hit the provider
    global UPSTREAM_CALLS
    UPSTREAM_CALLS += 1

    # Log request/response at informative level
    masked_req_headers = {
        k: (v[:4] + "..." if k.lower() == "x-app-token" else v)
        for k, v in resp.request.headers.items()
    }
    logger.info(
        "UPSTREAM CALLED → key=%s status=%s bytes≈%s token=%s req_headers=%s",
        key, resp.status_code, len(resp.content), _mask_token(APP_TOKEN), masked_req_headers
    )
    logger.debug("resp.headers = %s", dict(resp.headers))

    # 4) Handle result and populate cache on success
    if 200 <= resp.status_code < 300:
        data = resp.json()
        etag = resp.headers.get("ETag")
        last_modified = resp.headers.get("Last-Modified")
        _cache.set(key, data, etag=etag, last_modified=last_modified)
        return data

    # Non-success
    logger.warning("Failed upstream response for %s: HTTP %s", key, resp.status_code)
    return {"error": f"Failed to fetch data: {resp.status_code}"}


def cache_stats() -> dict:
    """Return cache metrics plus upstream call count."""
    stats = _cache.stats()
    stats["upstream_calls"] = UPSTREAM_CALLS
    stats["pid"] = PID
    stats["cache_id"] = id(_cache)
    return stats


def cache_clear() -> None:
    """Clear the in-process cache and reset the upstream call counter."""
    global UPSTREAM_CALLS
    _cache.clear()
    UPSTREAM_CALLS = 0
    logger.info("Cache cleared")
