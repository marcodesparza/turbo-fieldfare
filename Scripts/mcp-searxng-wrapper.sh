#!/bin/bash
set -euo pipefail

# Wrapper used by mcp.example.json to start the SearXNG MCP server.
# Set SEARXNG_DIR to the directory that contains start.sh and mcp-searxng.sh.
# Defaults to a sibling directory named "searxng" next to this checkout.

SEARXNG_DIR="${SEARXNG_DIR:-$(cd "$(dirname "$0")/.." && pwd)/../searxng}"

if [[ ! -d "$SEARXNG_DIR" ]]; then
    echo "Error: SearXNG directory not found at $SEARXNG_DIR" >&2
    echo "Set SEARXNG_DIR to the directory containing start.sh and mcp-searxng.sh" >&2
    exit 1
fi

exec "$SEARXNG_DIR/mcp-searxng.sh"
