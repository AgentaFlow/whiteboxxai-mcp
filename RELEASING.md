# Releasing

A published GitHub release triggers [`publish.yml`](.github/workflows/publish.yml), which
builds and publishes to PyPI, then to the MCP Registry, and optionally to Smithery.

## One-time setup

### PyPI (trusted publishing)

1. On pypi.org, go to **Account settings > Publishing** and add a **pending publisher**:
   - PyPI project name: `whiteboxxai-mcp`
   - Owner: `AgentaFlow`
   - Repository: `whiteboxxai-mcp`
   - Workflow: `publish.yml`
   - Environment: `pypi`
2. In this repository, go to **Settings > Environments** and create an environment named
   `pypi`. Optionally add required reviewers so a release waits for approval.

### MCP Registry

Nothing to configure. The `mcp-registry` job logs in with GitHub OIDC, which grants the
`io.github.AgentaFlow/*` namespace. The registry verifies PyPI ownership by looking for
`mcp-name: io.github.AgentaFlow/whiteboxxai-mcp` in the package README, so that marker must
stay in `README.md` (CI checks it).

To publish manually instead, install [`mcp-publisher`](https://github.com/modelcontextprotocol/registry/releases),
run `mcp-publisher login github` as an owner of the `AgentaFlow` org, then `mcp-publisher publish`.

### Smithery (optional, opt-in)

Smithery lists the hosted endpoint (`https://api.whiteboxxai.com/mcp/`), not the PyPI package.

1. Create or claim the `agentaflow` namespace on smithery.ai and generate an API key.
2. Add it as the secret `SMITHERY_API_KEY` in the `pypi` environment.
3. Set the repository variable `SMITHERY_PUBLISH` to `true`.
4. After the first publish, complete **Settings > Verification** on the Smithery server page.

Only enable this once the hosted endpoint accepts the `X-WhiteBox-API-Key` header declared in
[`smithery/config-schema.json`](smithery/config-schema.json); Smithery reserves `Authorization`
for its own OAuth, so the Bearer-only endpoint can't be used through it.

## Each release

1. Bump the version in all of these (CI's `server-json` job fails if they disagree):
   - `pyproject.toml`
   - `whiteboxxai_mcp/__init__.py`
   - `server.json` (`version` and `packages[0].version`)
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version.
3. If tools changed, regenerate the snapshot: `python tests/check_mcp_inspector_snapshot.py --write`.
4. Merge to `main`, then publish a GitHub release tagged `vX.Y.Z` (matching the version above;
   the workflow rejects a mismatch).
5. Verify:
   - <https://pypi.org/project/whiteboxxai-mcp/>
   - <https://registry.modelcontextprotocol.io/v0/servers?search=whiteboxxai>
