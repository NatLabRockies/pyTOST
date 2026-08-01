"""Minimal stdio MCP guidance server for pyTOST.

Implements the MCP stdio transport spec directly (JSON-RPC 2.0, one message
per line, no header framing -- see
https://modelcontextprotocol.io/specification/2025-06-18/basic/transports).
No third-party MCP SDK dependency is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pyTOST.mcp import content

SERVER_NAME = "pytost-mcp"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

TOOL_DEFS = [
    {
        "name": "overview",
        "description": "High-level overview of pyTOST: what it is, its engines, and when to use / not use it. Call first.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "engine_guide",
        "description": "How to choose among pyTOST's TOST engines (iid, cluster, temporal, spatial, spatiotemporal, building_aware).",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "strengths_and_weaknesses",
        "description": "Honest strengths/limitations of pyTOST.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_doc",
        "description": "Read a curated project doc from this repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": list(content.DOC_RESOURCES.keys()),
                    "description": "Documentation name.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
]


def _repo_root() -> Path:
    """Locate the repository root by walking up from this file to a pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


def _read_doc(repo_root: Path, name: str) -> str:
    relative_path = content.DOC_RESOURCES.get(name)
    if relative_path is None:
        raise ValueError(f"Unknown doc name: {name}")
    file_path = (repo_root / relative_path).resolve()
    if not file_path.exists():
        return f"Missing file: {relative_path}"
    return file_path.read_text(encoding="utf-8")


def dispatch(method: str, params: dict[str, Any] | None, repo_root: Path) -> dict[str, Any]:
    """Dispatch one MCP JSON-RPC method to its handler and return the result payload."""
    resolved_params = params or {}

    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }
    if method == "tools/list":
        return {"tools": TOOL_DEFS}
    if method == "tools/call":
        name = str(resolved_params.get("name", ""))
        arguments = resolved_params.get("arguments", {})
        if name == "overview":
            return {"content": [{"type": "text", "text": content.OVERVIEW}]}
        if name == "engine_guide":
            return {"content": [{"type": "text", "text": content.ENGINE_GUIDE}]}
        if name == "strengths_and_weaknesses":
            return {"content": [{"type": "text", "text": content.STRENGTHS_AND_WEAKNESSES}]}
        if name == "read_doc":
            doc_name = str(arguments.get("name", ""))
            return {"content": [{"type": "text", "text": _read_doc(repo_root, doc_name)}]}
        raise ValueError(f"Unknown tool: {name}")
    raise ValueError(f"Unsupported method: {method}")


def _read_message(stdin=None) -> dict[str, Any] | None:
    """Read one newline-delimited JSON-RPC message (MCP stdio transport spec)."""
    stream = stdin if stdin is not None else sys.stdin.buffer
    line = stream.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    return json.loads(line.decode("utf-8"))


def _write_message(message: dict[str, Any], stdout=None) -> None:
    stream = stdout if stdout is not None else sys.stdout.buffer
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=True)
    stream.write(payload.encode("utf-8"))
    stream.write(b"\n")
    stream.flush()


def main() -> int:
    """Run the stdio JSON-RPC loop until stdin is closed."""
    repo_root = _repo_root()
    while True:
        request = _read_message()
        if request is None:
            return 0
        if not request:
            continue

        method = request.get("method")
        req_id = request.get("id")
        params = request.get("params")

        if method == "notifications/initialized":
            continue
        if req_id is None:
            continue

        try:
            result = dispatch(str(method), params if isinstance(params, dict) else None, repo_root)
            _write_message({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception as exc:  # noqa: BLE001
            _write_message(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(exc)},
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
