"""Core implementation modules for AgenticRAGServerV2.

Everything in this package is plain library code: no FastMCP imports, no
`@mcp.tool` decorators, and **no side effects at import time**. Index/DB
loading happens in the FastMCP lifespan (`paper_fetch_mcp/app.py`), never
when a module is imported — see design spec §5 "Rules every module follows".

Thin `@mcp.tool` wrappers live in `paper_fetch_mcp/tools/` and call into here.
"""
