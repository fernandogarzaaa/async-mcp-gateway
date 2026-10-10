"""A real MCP stdio server built with the official `mcp` SDK.

Used as a demo backend for the gateway's /v1/mcp/{server} route. Each tenant
gets its own process (the gateway sets TENANT_ID), so notes are isolated.
"""

from __future__ import annotations

import os
from typing import Any

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

server: Any = _Server("notes")
_notes: list[str] = []


@server.tool()
def add_note(text: str) -> str:
    """Store a note for the calling tenant."""

    _notes.append(text)
    return f"saved note #{len(_notes)} for {os.getenv('TENANT_ID', '?')}"


@server.tool()
def list_notes() -> list[str]:
    """Return every note stored by the calling tenant."""

    return list(_notes)


if __name__ == "__main__":
    server.run("stdio")
