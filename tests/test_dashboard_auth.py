"""Tests for Dashboard API key authentication.

Covers:
- Property 1 (PBT): Unauthenticated requests are rejected on all protected routes
- Unit tests: dev mode (no key), valid key, wrong key, missing header

# Feature: watchdog-coverage-and-reliability, Property 1: Unauthenticated requests are rejected on all protected routes
"""

from __future__ import annotations

import importlib
import sys
import unittest.mock
from typing import Any, Optional, Tuple

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Protected routes under test
# ---------------------------------------------------------------------------

PROTECTED_ROUTES: list[tuple[str, str]] = [
    ("GET", "/dashboard"),
    ("GET", "/api/findings"),
    ("GET", "/api/findings/test-id"),
    ("POST", "/api/findings/test-id/approve"),
    ("POST", "/api/findings/test-id/decline"),
    ("GET", "/api/runs"),
    ("GET", "/api/runs/test-run-id"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_KEY = "super-secret-key-123"


def _make_client(api_key: str) -> tuple["TestClient", Any]:
    """Import dashboard with DASHBOARD_API_KEY patched to *api_key*.

    We reload the module each time so the module-level ``DASHBOARD_API_KEY``
    and ``_bearer_scheme`` are re-evaluated with the patched env var.

    Returns (client, dash_mod) so callers can patch the module variable if needed.
    """
    # Patch the env var before importing/reloading the module.
    with unittest.mock.patch.dict("os.environ", {"DASHBOARD_API_KEY": api_key}, clear=False):
        # Force a fresh import so module-level code re-runs.
        if "migration_watchdog.dashboard" in sys.modules:
            del sys.modules["migration_watchdog.dashboard"]
        import migration_watchdog.dashboard as dash_mod  # noqa: PLC0415

    return TestClient(dash_mod.app, raise_server_exceptions=False), dash_mod


def _bearer_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# Property 1 (PBT): Unauthenticated requests are rejected on all protected routes
# Validates: Requirements 1.1, 1.2, 1.5
# ---------------------------------------------------------------------------

# Feature: watchdog-coverage-and-reliability, Property 1: Unauthenticated requests are rejected on all protected routes
@given(
    route=st.sampled_from(PROTECTED_ROUTES),
    bad_credential=st.one_of(
        st.none(),                                              # no Authorization header
        st.just(""),                                            # empty header value
        st.text(                                                # wrong key (printable ASCII, not equal to _VALID_KEY)
            alphabet=st.characters(min_codepoint=33, max_codepoint=126),
            min_size=1,
            max_size=64,
        ).filter(lambda s: s != _VALID_KEY),
    ),
)
@settings(max_examples=100, deadline=None)
def test_unauthenticated_requests_rejected(
    route: tuple[str, str],
    bad_credential: str | None,
) -> None:
    """Property 1: Any request to a protected route with a missing or wrong key
    must return HTTP 401 when DASHBOARD_API_KEY is set.

    Validates: Requirements 1.1, 1.2, 1.5
    """
    method, path = route
    client, dash_mod = _make_client(_VALID_KEY)

    if bad_credential is None:
        # No Authorization header at all.
        headers: dict[str, str] = {}
    elif bad_credential == "":
        # Empty Authorization header — not a valid Bearer token.
        headers = {"Authorization": ""}
    else:
        headers = _bearer_headers(bad_credential)

    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", _VALID_KEY):
        resp = client.request(method, path, headers=headers)
    assert resp.status_code == 401, (
        f"Expected 401 for {method} {path} with credential={bad_credential!r}, "
        f"got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# Unit tests: edge cases
# ---------------------------------------------------------------------------


def test_dev_mode_allows_all_requests() -> None:
    """Req 1.3: When DASHBOARD_API_KEY is empty, all requests are allowed through."""
    client, dash_mod = _make_client("")
    # /health is always open; /api/runs requires auth when key is set.
    # In dev mode it should NOT return 401.
    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", ""):
        resp = client.get("/api/runs")
    assert resp.status_code != 401, (
        f"Expected non-401 in dev mode, got {resp.status_code}"
    )


def test_valid_bearer_key_returns_non_401() -> None:
    """Req 1.4: A correct Authorization: Bearer <key> header is accepted."""
    client, dash_mod = _make_client(_VALID_KEY)
    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", _VALID_KEY):
        resp = client.get("/api/runs", headers=_bearer_headers(_VALID_KEY))
    # The route may fail for other reasons (e.g. no DynamoDB), but it must
    # not return 401.
    assert resp.status_code != 401, (
        f"Expected non-401 for valid key, got {resp.status_code}"
    )


def test_wrong_key_returns_401() -> None:
    """Req 1.1: A wrong key in the Authorization header returns 401."""
    client, dash_mod = _make_client(_VALID_KEY)
    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", _VALID_KEY):
        resp = client.get("/api/runs", headers=_bearer_headers("wrong-key"))
    assert resp.status_code == 401


def test_missing_authorization_header_returns_401() -> None:
    """Req 1.1: A missing Authorization header returns 401 when key is set."""
    client, dash_mod = _make_client(_VALID_KEY)
    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", _VALID_KEY):
        resp = client.get("/api/runs")
    assert resp.status_code == 401


def test_health_endpoint_is_unauthenticated() -> None:
    """The /health endpoint must remain open regardless of DASHBOARD_API_KEY."""
    client, dash_mod = _make_client(_VALID_KEY)
    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", _VALID_KEY):
        resp = client.get("/health")
    assert resp.status_code == 200


def test_all_protected_routes_reject_missing_key() -> None:
    """Req 1.5: Every protected route returns 401 when the key is missing."""
    client, dash_mod = _make_client(_VALID_KEY)
    with unittest.mock.patch.object(dash_mod, "DASHBOARD_API_KEY", _VALID_KEY):
        for method, path in PROTECTED_ROUTES:
            resp = client.request(method, path)
            assert resp.status_code == 401, (
                f"Expected 401 for {method} {path} with no key, got {resp.status_code}"
            )
