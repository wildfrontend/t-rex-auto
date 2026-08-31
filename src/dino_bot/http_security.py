"""Shared validation for the bot's loopback-only HTTP control surfaces."""

from __future__ import annotations

from urllib.parse import urlparse

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_loopback_origin(origin: str) -> bool:
    """Allow browser loopback origins and non-browser clients without Origin."""

    if not origin:
        return True
    parsed = urlparse(origin)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in _LOOPBACK_HOSTS
        and parsed.username is None
        and parsed.password is None
    )
