"""Standalone zero-dependency stdio JSON-RPC MCP test server."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

JsonDict = dict[str, Any]


def main() -> int:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            write_response(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "parse error",
                        "data": str(exc),
                    },
                }
            )
            continue

        response = handle_request(request)
        if response is not None:
            write_response(response)
    return 0


def handle_request(request: JsonDict) -> JsonDict | None:
    request_id = request.get("id")
    method = request.get("method")
    raw_params = request.get("params")
    params: JsonDict = raw_params if isinstance(raw_params, dict) else {}

    if request.get("jsonrpc") != "2.0" or not isinstance(method, str):
        return error_response(request_id, -32600, "invalid request")

    delay = float(params.get("delay_seconds", 0) or 0)
    if delay > 0:
        time.sleep(delay)

    if method == "ping":
        return success_response(
            request_id, {"status": "ok", "tenant_id": os.getenv("TENANT_ID", "")}
        )

    if method == "tools/list":
        return success_response(
            request_id,
            {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echoes arguments back to the caller.",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "delay",
                        "description": "Sleeps for delay_seconds before responding.",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "error",
                        "description": "Returns a JSON-RPC application error.",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "hang",
                        "description": "Sleeps forever to validate timeout and reaping behavior.",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "crash",
                        "description": "Terminates the server process.",
                        "inputSchema": {"type": "object"},
                    },
                ]
            },
        )

    if method == "tools/call":
        tool_name = str(params.get("name", "echo"))
        raw_arguments = params.get("arguments")
        arguments: JsonDict = raw_arguments if isinstance(raw_arguments, dict) else {}
        if tool_name == "delay":
            time.sleep(float(arguments.get("seconds", 0.1)))
            return success_response(
                request_id, {"content": [{"type": "text", "text": "delayed"}]}
            )
        if tool_name == "error":
            return error_response(
                request_id, -32001, "simulated tool failure", arguments
            )
        if tool_name == "hang":
            while True:
                time.sleep(3600)
        if tool_name == "crash":
            sys.stderr.write("mock_mcp_server crashing on request\n")
            sys.stderr.flush()
            os._exit(17)
        return success_response(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "tool": tool_name,
                                "arguments": arguments,
                                "tenant_id": os.getenv("TENANT_ID", ""),
                            },
                            sort_keys=True,
                        ),
                    }
                ]
            },
        )

    if method == "notifications/initialized":
        return None

    return error_response(request_id, -32601, f"method not found: {method}")


def success_response(request_id: object, result: JsonDict) -> JsonDict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(
    request_id: object, code: int, message: str, data: object = None
) -> JsonDict:
    error: JsonDict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def write_response(response: JsonDict) -> None:
    sys.stdout.write(
        json.dumps(response, separators=(",", ":"), ensure_ascii=False) + "\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
