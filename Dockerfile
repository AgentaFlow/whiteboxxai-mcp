# WhiteBoxXAI MCP Server -- minimal stdio image. Installs only this package and
# its PyPI dependencies (including whitebox-xai-sdk), with no backend or ML
# dependencies.
#
# Build from the repo root:
#   docker build -t whiteboxxai-mcp .
#
# Run (stdio transport -- an MCP client spawns/attaches to this process's
# stdin/stdout; it is not a long-running network service, so no port is
# exposed and nothing here should be deployed as an always-on host):
#   docker run --rm -i -e WHITEBOXXAI_MCP_API_KEY=wbx_live_... \
#       -e WHITEBOXXAI_MCP_API_BASE_URL=https://api.whiteboxxai.com \
#       whiteboxxai-mcp

FROM python:3.11-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd -m -u 1000 -s /bin/bash appuser

# ===========================================================================
# Builder -- installs this package into site-packages only.
# ===========================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./src/
COPY whiteboxxai_mcp/ ./src/whiteboxxai_mcp/

RUN pip install --no-cache-dir ./src

# ===========================================================================
# Runtime -- slim image, no compiler toolchain, non-root user.
# ===========================================================================
FROM base AS runtime

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

USER appuser

ENTRYPOINT ["whiteboxxai-mcp"]
