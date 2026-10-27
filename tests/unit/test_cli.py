"""Tests for the `whiteboxxai-mcp` CLI entry point's credential gate.

No live WhiteBoxXAI backend or stdio loop is exercised -- `build_server` and
the returned server's `.run()` are mocked so these only exercise the
credential validation in whiteboxxai_mcp/cli.py:main().
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from whiteboxxai_mcp import cli


def _settings(**overrides):
    base = {
        "API_KEY": None,
        "JWT": None,
        "EMAIL": None,
        "PASSWORD": None,
        "LOG_LEVEL": "INFO",
        "VERBOSE": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _no_argv():
    with patch("sys.argv", ["whiteboxxai-mcp"]):
        yield


def _run_main_with(settings_obj):
    fake_server = MagicMock()
    with patch.object(cli, "settings", settings_obj), patch.object(
        cli, "build_server", return_value=fake_server
    ) as build, patch.object(cli, "configure_logging"):
        cli.main()
    return build, fake_server


class TestCredentialGate:
    def test_api_key_only_is_accepted(self):
        build, fake_server = _run_main_with(_settings(API_KEY="wbx_live_abc"))
        build.assert_called_once()
        fake_server.run.assert_called_once_with(transport="stdio")

    def test_no_credentials_exits_with_code_2(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_main_with(_settings())
        assert exc_info.value.code == 2
