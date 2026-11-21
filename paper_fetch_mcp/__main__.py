"""`python -m paper_fetch_mcp` -- the server entrypoint.

All loading (FTS init, publisher cache table, jobs/telemetry DBs) happens
inside `app.build()`'s FastMCP lifespan, never at import time here.
"""

from __future__ import annotations

import argparse

from paper_fetch_mcp.app import build


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperFetchServer MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "streamable-http", "sse"], default="stdio"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    server = build()

    if args.transport == "stdio":
        server.run(transport=args.transport)
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
