"""Shared connectivity check used to route online vs offline behaviour."""

from __future__ import annotations

import socket
import time

_cache: dict[str, tuple[float, bool]] = {}
_TTL_SECONDS = 15.0


def is_online(host: str = "1.1.1.1", port: int = 53, timeout: float = 1.5) -> bool:
    """Best-effort internet check via a fast TCP connect, cached briefly.

    Uses a DNS resolver's TCP port rather than an HTTP request so it stays
    quick and doesn't depend on any particular site being up. Cached for a
    few seconds so a voice turn doesn't probe the network repeatedly.
    """
    now = time.monotonic()
    cached = _cache.get("online")
    if cached and now - cached[0] < _TTL_SECONDS:
        return cached[1]

    result = False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            result = True
    except OSError:
        result = False

    _cache["online"] = (now, result)
    return result
