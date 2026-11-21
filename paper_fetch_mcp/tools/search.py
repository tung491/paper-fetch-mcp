"""Search tools: search_metadata.

Thin `@mcp.tool` wrapper over `core.papers`. Registered on the shared
`paper_fetch_mcp.mcp` instance; imported by `paper_fetch_mcp.app.build()`,
never at any other module's import time.

This server has no semantic or keyword search over paper text -- metadata
search over the papers database is the only local search.
"""

from __future__ import annotations

import logging
from typing import Optional

from paper_fetch_mcp import mcp
from paper_fetch_mcp.core import output
from paper_fetch_mcp.core import papers

logger = logging.getLogger(__name__)


@mcp.tool(timeout=45)
def search_metadata(
    query: Optional[str] = None,
    author: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 20,
) -> str:
    """
    Unified metadata search over the papers database: one tool that
    dispatches on whichever filters are given.

    Args:
        query: Free-text search across title/abstract/author (FTS5 when
            initialised, else LIKE).
        author: Author name substring filter.
        year_from: Inclusive lower publication-year bound.
        year_to: Inclusive upper publication-year bound.
        limit: Maximum results (1-100, default 20).

    Returns:
        YAML with the matching rows, the filters that were applied, and a
        count. An error when none of query/author/year_from/year_to is
        given.
    """
    limit = min(max(limit, 1), 100)
    result = papers.search_metadata(
        query=query,
        author=author,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
    )
    return output.respond(result, list_key="results")
