"""PaperFetchServer MCP package."""

from fastmcp import FastMCP

_INSTRUCTIONS = (
    "PaperFetchServer: full-text access to research papers. `get_paper` "
    "resolves a citation key, DOI, arXiv id or title against the local "
    "paper corpus and fetches the paper online when it is not there. Also "
    "paper metadata search, Semantic Scholar and alphaXiv lookups, "
    "publisher-page fetching, arXiv ingestion and BibTeX helpers.\n\n"
    "There is no semantic or keyword search over paper text: find papers "
    "with `search_metadata` (local corpus) or `semantic_scholar_search` "
    "(online), then read them with `get_paper`."
)

# Constructed with static instructions only -- no disk/network I/O at
# package-import time (see core/__init__.py's "no side effects at import"
# rule). Tools are registered on this instance by `paper_fetch_mcp.app.build()`.
mcp = FastMCP("PaperFetchServer", instructions=_INSTRUCTIONS)
