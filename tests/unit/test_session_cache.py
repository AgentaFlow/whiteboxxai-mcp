"""Tests for the per-connection HTTP session cache (issue #226).

No live backend or network calls -- resolve_http_session() only builds lazy
static-JWT SessionManager instances (SessionManager itself never touches the
network until get_client() is actually called), so this exercises the
cache's keying/reuse/eviction logic in isolation.
"""

import time

import pytest

from whiteboxxai_mcp.auth import context as ctx


@pytest.fixture(autouse=True)
def _reset_state():
    """Every test starts from a clean cache and stdio singleton."""
    ctx.reset_http_session_cache()
    ctx.reset_stdio_session()
    yield
    ctx.reset_http_session_cache()
    ctx.reset_stdio_session()


class TestNoCrossKeyLeakage:
    def test_different_tokens_resolve_to_different_sessions(self):
        session_a = ctx.resolve_http_session("https://api.test.com", "key-org-a")
        session_b = ctx.resolve_http_session("https://api.test.com", "key-org-b")

        assert session_a is not session_b
        assert session_a._api_key == "key-org-a"
        assert session_b._api_key == "key-org-b"

    def test_cache_keys_are_hashed_not_the_raw_token(self):
        token = "jwt-org-a"
        ctx.resolve_http_session("https://api.test.com", token)

        assert token not in ctx._http_session_cache
        assert ctx._hash_token(token) in ctx._http_session_cache


class TestReuseWithinTTL:
    def test_same_token_resolved_twice_returns_same_instance(self):
        session1 = ctx.resolve_http_session("https://api.test.com", "jwt-1")
        session2 = ctx.resolve_http_session("https://api.test.com", "jwt-1")

        assert session1 is session2

    def test_resolving_a_second_token_does_not_evict_the_first(self):
        session1 = ctx.resolve_http_session("https://api.test.com", "jwt-1")
        ctx.resolve_http_session("https://api.test.com", "jwt-2")

        assert ctx.resolve_http_session("https://api.test.com", "jwt-1") is session1


class TestTTLEviction:
    def test_expired_entry_is_rebuilt_not_reused(self, monkeypatch):
        monkeypatch.setattr(ctx.settings, "HTTP_SESSION_CACHE_TTL_SECONDS", 1)
        session1 = ctx.resolve_http_session("https://api.test.com", "jwt-1")

        fake_now = time.monotonic() + 1000

        def fake_monotonic():
            return fake_now

        monkeypatch.setattr(ctx.time, "monotonic", fake_monotonic)
        session2 = ctx.resolve_http_session("https://api.test.com", "jwt-1")

        # No error, and a genuinely new instance -- eviction is side-effect-free.
        assert session2 is not session1
        assert session2._api_key == "jwt-1"

    def test_access_within_ttl_refreshes_last_access(self, monkeypatch):
        monkeypatch.setattr(ctx.settings, "HTTP_SESSION_CACHE_TTL_SECONDS", 100)
        session1 = ctx.resolve_http_session("https://api.test.com", "jwt-1")
        key = ctx._hash_token("jwt-1")
        _, first_access = ctx._http_session_cache[key]

        later = first_access + 50

        monkeypatch.setattr(ctx.time, "monotonic", lambda: later)
        session2 = ctx.resolve_http_session("https://api.test.com", "jwt-1")

        assert session2 is session1
        _, second_access = ctx._http_session_cache[key]
        assert second_access == later


class TestOpportunisticSweep:
    def test_sweep_clears_stale_entries_when_a_fresh_token_is_resolved(
        self, monkeypatch
    ):
        monkeypatch.setattr(ctx.settings, "HTTP_SESSION_CACHE_TTL_SECONDS", 10)
        monkeypatch.setattr(
            ctx.settings, "HTTP_SESSION_CACHE_SWEEP_INTERVAL_SECONDS", 5
        )

        t0 = time.monotonic()
        monkeypatch.setattr(ctx.time, "monotonic", lambda: t0)
        ctx.resolve_http_session("https://api.test.com", "jwt-stale-1")
        ctx.resolve_http_session("https://api.test.com", "jwt-stale-2")
        assert len(ctx._http_session_cache) == 2

        # Past both the TTL and the sweep interval.
        t1 = t0 + 20
        monkeypatch.setattr(ctx.time, "monotonic", lambda: t1)
        ctx.resolve_http_session("https://api.test.com", "jwt-fresh")

        remaining = set(ctx._http_session_cache.keys())
        assert ctx._hash_token("jwt-stale-1") not in remaining
        assert ctx._hash_token("jwt-stale-2") not in remaining
        assert ctx._hash_token("jwt-fresh") in remaining


class TestGetSessionDispatch:
    def test_falls_back_to_stdio_singleton_when_contextvar_unset(self, monkeypatch):
        # ctx.settings is the module-level MCPSettings singleton (already
        # constructed at import time, so setenv after the fact wouldn't be
        # read) -- set the attribute it reads directly instead.
        monkeypatch.setattr(ctx.settings, "API_KEY", "stdio-api-key")
        stdio_session = ctx.get_stdio_session()

        assert ctx.get_session() is stdio_session

    def test_uses_active_http_session_when_contextvar_set(self):
        http_session = ctx.resolve_http_session("https://api.test.com", "jwt-http")
        token = ctx._active_http_session.set(http_session)
        try:
            assert ctx.get_session() is http_session
        finally:
            ctx._active_http_session.reset(token)
