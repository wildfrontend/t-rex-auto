from __future__ import annotations

import pytest

from dino_bot.http_security import is_loopback_origin


@pytest.mark.parametrize(
    "origin",
    ["", "http://localhost", "https://127.0.0.1:8780", "http://[::1]:8765"],
)
def test_loopback_origin_allowlist(origin: str) -> None:
    assert is_loopback_origin(origin)


@pytest.mark.parametrize(
    "origin",
    [
        "https://example.com",
        "http://localhost.example.com",
        "ftp://localhost",
        "http://user@localhost",
        "not a URL",
    ],
)
def test_loopback_origin_rejects_remote_lookalike_and_userinfo(origin: str) -> None:
    assert not is_loopback_origin(origin)
