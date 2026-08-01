# pyTOST MCP server

pyTOST ships a minimal, dependency-free MCP (Model Context Protocol)
guidance server so MCP-capable agents/editors (e.g. GitHub Copilot CLI) can
discover *when* and *how* to use pyTOST without reading the source tree.

This is a documentation/guidance server, not a code-execution server: it
exposes read-only tools that return curated text (overview, engine-selection
guidance, strengths/weaknesses, and project docs). It does not run pyTOST
computations, evaluate arbitrary code, or write to the repository.

## Why this exists

pyTOST is intentionally library-first (see `CONTRIBUTING.md`); this MCP
server was added on explicit request as a narrow, additive integration and
does not change the package's public API, add a CLI, or alter existing
packaging.

## Transport

The server implements the MCP stdio transport directly: JSON-RPC 2.0
messages, one per line, newline-delimited (**not** LSP-style
`Content-Length` header framing — see the
[MCP stdio transport spec](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)).
No third-party MCP SDK is required.

## Tools

- `overview` — what pyTOST is, when to use / not use it.
- `engine_guide` — how to choose among the `iid`, `cluster`, `temporal`,
  `spatial`, `spatiotemporal`, and `building_aware` engines.
- `strengths_and_weaknesses` — honest strengths/limitations.
- `read_doc` — read a curated project doc (`readme`, `contributing`) by name.

## Running it directly

```bash
pixi run python -m pyTOST.mcp
```

Reads JSON-RPC requests from stdin, writes responses to stdout, one message
per line.

## Registering with GitHub Copilot CLI

pyTOST has no `pixi` on `PATH` when spawned by Copilot CLI (which sanitizes
`PATH` for child MCP processes) and needs its working directory set to the
repo root for module resolution, so register it via a `bash -c 'cd ... &&
exec pixi run ...'` wrapper with an absolute path to `pixi`:

```bash
copilot mcp add pytost -- /bin/bash -c \
  "cd /path/to/pyTOST && exec /path/to/.pixi/bin/pixi run python -m pyTOST.mcp"
```

Verify with:

```bash
copilot mcp get pytost
copilot mcp list
```

## Tests

```bash
pixi run -e test python -m pytest tests/test_mcp_server.py -q
```
