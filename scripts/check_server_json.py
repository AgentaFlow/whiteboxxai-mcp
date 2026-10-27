"""Validate server.json and check that every version string in the repo agrees.

Checks:
  * server.json validates against the MCP Registry schema it declares in `$schema`.
  * server.json `version`, `packages[*].version`, pyproject.toml `version`, and
    `whiteboxxai_mcp.__version__` are identical.
  * README.md carries the `mcp-name:` marker matching server.json `name`, which
    the MCP Registry uses to verify PyPI package ownership.

Usage:
    python scripts/check_server_json.py            # validates against the live schema
    python scripts/check_server_json.py --offline  # skip schema validation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _match_version(pattern: str, text: str, label: str) -> str:
    found = re.search(pattern, text, flags=re.MULTILINE)
    if not found:
        raise SystemExit(f"could not find a version in {label}")
    return found.group(1)


def _validate_schema(doc: dict) -> list[str]:
    from jsonschema import validators

    with urllib.request.urlopen(doc["$schema"], timeout=30) as resp:  # nosec B310
        schema = json.load(resp)
    validator = validators.validator_for(schema)(schema)
    return [
        f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
        for e in validator.iter_errors(doc)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="skip schema validation")
    args = parser.parse_args()

    doc = json.loads(_read("server.json"))
    problems: list[str] = [] if args.offline else _validate_schema(doc)

    versions = {
        "server.json version": doc["version"],
        "pyproject.toml version": _match_version(
            r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), "pyproject.toml"
        ),
        "whiteboxxai_mcp.__version__": _match_version(
            r'^__version__\s*=\s*"([^"]+)"',
            _read("whiteboxxai_mcp/__init__.py"),
            "__init__.py",
        ),
    }
    for i, package in enumerate(doc.get("packages", [])):
        versions[f"server.json packages[{i}].version"] = package["version"]
    if len(set(versions.values())) != 1:
        problems.append(
            "version mismatch: "
            + ", ".join(f"{k}={v}" for k, v in sorted(versions.items()))
        )

    marker = f"mcp-name: {doc['name']}"
    if not re.search(re.escape(marker) + r"(\s|-->|$)", _read("README.md")):
        problems.append(f"README.md is missing the registry marker '{marker}'")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"OK: server.json valid, all versions = {doc['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
