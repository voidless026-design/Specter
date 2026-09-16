"""Remembering what isn't available, so we stop asking.

Loading a model that was never downloaded costs a network timeout, and the
CLI is a fresh process every time - so `ev search` on a machine with no
internet paid five seconds to rediscover the same "no" on every run. That is
the exact machine E.V. is meant to work on.

So a negative probe is written to a small JSON file next to the store and
honoured for an hour. Positive results are not cached: a model that loaded
once will load again quickly from disk, and caching "yes" would only create
a way to be wrong.

The TTL is short enough that installing a model is noticed without anyone
having to know this file exists, and `forget()` clears it outright - which
is what `ev doctor` does, since the whole point of a health check is a live
answer.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

TTL_SECONDS = 3600.0
FILENAME = "probe-cache.json"


def _path(cfg) -> Path | None:
    data_dir = getattr(cfg, "data_dir", None)
    return Path(data_dir) / FILENAME if data_dir else None


def _read(cfg) -> dict:
    path = _path(cfg)
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def recently_failed(cfg, key: str, ttl: float = TTL_SECONDS) -> bool:
    """Whether `key` failed to load recently enough not to bother retrying."""
    entry = _read(cfg).get(key)
    if not isinstance(entry, (int, float)):
        return False
    return (time.time() - entry) < ttl


def remember_failure(cfg, key: str) -> None:
    """Note that `key` couldn't be loaded, so the next process skips it."""
    path = _path(cfg)
    if path is None:
        return
    entries = _read(cfg)
    entries[key] = time.time()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries), encoding="utf-8")
    except OSError as e:
        logger.debug("Couldn't write the probe cache: %s", e)


def forget(cfg, key: str | None = None) -> None:
    """Clear one remembered failure, or all of them."""
    path = _path(cfg)
    if path is None:
        return
    if key is None:
        path.unlink(missing_ok=True)
        return
    entries = _read(cfg)
    if entries.pop(key, None) is not None:
        try:
            path.write_text(json.dumps(entries), encoding="utf-8")
        except OSError:
            pass
