"""Minimal MCP client that talks to a stdio MCP server *through* the gateway.

Example:
    uv run python scripts/mcp_http_client.py --server notes \\
        --call add_note '{"text": "hello"}' --call list_notes '{}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument("--tenant", default=os.getenv("GATEWAY_TENANT", "tenant-alpha"))
    parser.add_argument(
        "--token", default=os.getenv("GATEWAY_TOKEN", "alpha-secret-token")
    )
    parser.add_argument("--server", required=True, help="MCP server name")
    parser.add_argument(
        "--call",
        nargs=2,
        action="append",
        default=[],
        metavar=("TOOL", "JSON_ARGS"),
        help="tool to call after listing tools (repeatable)",
    )
    args = parser.parse_args(argv)

    headers = {"X-Tenant-ID": args.tenant, "Authorization": f"Bearer {args.token}"}
    url = f"{args.base_url.rstrip('/')}/v1/mcp/{args.server}"
    with httpx.Client(headers=headers, timeout=60, trust_env=False) as client:
        counter = 0

        def rpc(method: str, params: dict[str, Any]) -> Any:
            nonlocal counter
            counter += 1
            body = {"jsonrpc": "2.0", "id": counter, "method": method}
            body["params"] = params
            response = client.post(url, json=body)
            if response.status_code != 200:
                raise SystemExit(
                    f"{method}: HTTP {response.status_code} {response.text}"
                )
            payload = response.json()
            if "error" in payload:
                raise SystemExit(f"{method}: JSON-RPC error {payload['error']}")
            return payload["result"]

        init = rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "gateway-http-client", "version": "0.1"},
            },
        )
        print("initialize ->", json.dumps(init.get("serverInfo", init)))
        notified = client.post(
            url, json={"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        print("notifications/initialized -> HTTP", notified.status_code)
        tools = rpc("tools/list", {})
        print("tools/list ->", [tool["name"] for tool in tools.get("tools", [])])
        for name, raw_args in args.call:
            result = rpc(
                "tools/call", {"name": name, "arguments": json.loads(raw_args)}
            )
            texts = [c.get("text") for c in result.get("content", []) if "text" in c]
            print(f"tools/call {name} ->", texts or result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
