"""Tests for ASGI authentication extraction (issue #930).

Verifies extraction priority between X-WhiteBox-API-Key and Authorization: Bearer,
header normalization, and rejection of invalid / missing credentials.
"""

from whiteboxxai_mcp.asgi import _extract_auth_token, _extract_bearer_token


class TestExtractAuthToken:
    def test_extract_from_api_key_header(self):
        scope = {
            "type": "http",
            "headers": [
                (b"x-whitebox-api-key", b"wbx_live_custom_key_123"),
            ],
        }
        assert _extract_auth_token(scope) == "wbx_live_custom_key_123"

    def test_extract_from_authorization_bearer(self):
        scope = {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer eyJhbGciOiJIUzI1NiJ9.token"),
            ],
        }
        assert _extract_auth_token(scope) == "eyJhbGciOiJIUzI1NiJ9.token"

    def test_api_key_header_takes_precedence_over_authorization(self):
        scope = {
            "type": "http",
            "headers": [
                (b"x-whitebox-api-key", b"wbx_live_preferred_key"),
                (b"authorization", b"Bearer gateway_proxy_oauth_token"),
            ],
        }
        assert _extract_auth_token(scope) == "wbx_live_preferred_key"

    def test_missing_headers_returns_none(self):
        scope = {"type": "http", "headers": []}
        assert _extract_auth_token(scope) is None

    def test_non_bearer_authorization_returns_none(self):
        scope = {
            "type": "http",
            "headers": [
                (b"authorization", b"Basic dXNlcjpwYXNz"),
            ],
        }
        assert _extract_auth_token(scope) is None

    def test_empty_tokens_return_none(self):
        assert (
            _extract_auth_token(
                {"type": "http", "headers": [(b"x-whitebox-api-key", b"   ")]}
            )
            is None
        )
        assert (
            _extract_auth_token(
                {"type": "http", "headers": [(b"authorization", b"Bearer   ")]}
            )
            is None
        )

    def test_extract_bearer_token_backwards_compatibility_alias(self):
        assert _extract_bearer_token is _extract_auth_token
