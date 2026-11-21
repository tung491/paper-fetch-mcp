"""External search tools: semantic_scholar_search, alphaxiv_paper_lookup,
alphaxiv_batch_lookup, fetch_publisher_html.

Thin `@mcp.tool` wrappers over `core.s2` / `core.alphaxiv` / `core.publisher`
(design spec
`docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md`, §5).
This REPLACES the previous dead-fork `tools/external.py`, which had its own
naive `requests.get` with no rate limiter/retry/cache -- `core.s2` is the
Phase 0 fix for that (finding F4: 29 silent HTTP 429s and 50 hard errors
mined from a month of transcripts).
"""

from __future__ import annotations

import logging
from typing import List

from paper_fetch_mcp import mcp
from paper_fetch_mcp.core import alphaxiv
from paper_fetch_mcp.core import output
from paper_fetch_mcp.core import publisher
from paper_fetch_mcp.core import s2

logger = logging.getLogger(__name__)


@mcp.tool(timeout=45)
def semantic_scholar_search(
    query: str,
    venue: str = "",
    year: str = "",
    fields_of_study: str = "",
    limit: int = 10,
) -> str:
    """
    Search for papers using the Semantic Scholar API.

    Args:
        query: Search query string.
        venue: Optional venue filter. Must be an EXACT venue name as
            Semantic Scholar stores it (e.g., "IEEE Transactions on
            Wireless Communications", "Neural Information Processing
            Systems"). Comma-separated for multiple venues. Substrings
            like "IEEE" will return zero results.
        year: Optional year filter -- a single year ("2023") or a range
            ("2019-2023", "2019-", "-2023").
        fields_of_study: Optional comma-separated field-of-study filter
            (e.g. "Computer Science,Engineering").
        limit: Number of results (1-100, default 10).

    Returns:
        YAML with compact paper rows (title, year, venue, doi, arxiv_id,
        citation_count, abstract truncated to 400 chars) each carrying
        `in_corpus: <citation_key>` when that paper is already in the local
        corpus (matched by DOI, then arXiv id, then title), or `null` when
        it is not -- use that to pivot straight to
        `get_content_by_citation_key` instead of another Semantic Scholar
        call.
    """
    result = s2.search(
        query, venue=venue, year=year, fields_of_study=fields_of_study, limit=limit
    )
    if not result.get("ok"):
        return output.error(
            f"Semantic Scholar search failed: {result.get('error')}",
            kind="upstream_error",
        )
    data = result["data"]
    return output.respond(
        {"total": data.get("total"), "papers": data.get("papers", [])},
        list_key="papers",
    )


@mcp.tool(timeout=45)
def semantic_scholar_paper(identifier: str) -> str:
    """
    Look up a single paper on Semantic Scholar by DOI, arXiv id, or
    Semantic Scholar paper id.

    Pairs with `doi2content` misses (spec §6): when a paper isn't in the
    local corpus, this still gets its metadata and abstract from Semantic
    Scholar. Shares `semantic_scholar_search`'s rate limiter, retry, and
    cache.

    Args:
        identifier: A DOI (e.g. "10.1109/TWC.2024.001"), an arXiv id (e.g.
            "2301.12345", "arXiv:2301.12345"), or a Semantic Scholar paper
            id.

    Returns:
        YAML with the same compact shape as `semantic_scholar_search` rows
        (title, year, venue, doi, arxiv_id, citation_count, abstract<=400
        chars, in_corpus).
    """
    result = s2.get_paper(identifier)
    if not result.get("ok"):
        return output.error(
            f"Semantic Scholar paper lookup failed: {result.get('error')}",
            kind="upstream_error",
        )
    return output.respond(result["data"])


@mcp.tool(timeout=45)
def alphaxiv_paper_lookup(paper_id_or_url: str, max_chars: int = 15000) -> str:
    """
    Look up any ArXiv paper via AlphaXiv.org to get a structured
    AI-generated overview -- faster and more reliable than reading a raw
    PDF.

    Supports ArXiv URLs, PDF URLs, AlphaXiv URLs, bare IDs, and versioned
    IDs. Returns the machine-readable overview (markdown). Falls back to
    full text if the overview endpoint 404s. Returns an error if both
    endpoints fail.

    Args:
        paper_id_or_url: An ArXiv ID, versioned ID, or URL.
        max_chars: Truncate the fetched overview/full_text to this many
            characters (default 15000).

    Returns:
        YAML with paper_id, overview/full_text, source, and status fields.
    """
    result = alphaxiv.lookup(paper_id_or_url, max_chars=max_chars)
    return output.respond(result)


@mcp.tool(timeout=45)
def alphaxiv_batch_lookup(paper_ids_or_urls: List[str], max_chars: int = 2000) -> str:
    """
    Look up multiple ArXiv papers via AlphaXiv.org in a single call.

    For each ID/URL, fetches the machine-readable overview (falls back to
    full text on 404). Useful when the user shares several papers at once.

    Args:
        paper_ids_or_urls: List of ArXiv IDs, versioned IDs, or URLs (max
            20).
        max_chars: Truncate each paper's overview/full_text to this many
            characters (default 2000 -- batch responses overflowed 100% of
            the time before this limit existed).

    Returns:
        YAML with a `results` list, one entry per paper_id, each carrying
        its own status/overview/full_text/error fields.
    """
    MAX = 20
    paper_ids_or_urls = paper_ids_or_urls[:MAX]
    results = []
    for raw in paper_ids_or_urls:
        item = alphaxiv.lookup(raw, max_chars=max_chars)
        item.setdefault("paper_id", raw)
        results.append(item)
    return output.respond({"results": results}, list_key="results")


# 90 s like get_paper: requests, then a headless-browser render on this
# host, then the exit node -- each stage can take 20-30 s.
@mcp.tool(timeout=90)
def fetch_publisher_html(identifier: str, force_refresh: bool = False) -> str:
    """
    Fetch a paper's HTML content from its publisher's website (IEEE,
    Springer, ScienceDirect, etc.) via a running Chrome instance, falling
    back to the Semantic Scholar abstract if the browser fetch fails.

    The result is cached in papers.db. Pass force_refresh=True to bypass
    the cache.

    Args:
        identifier: A DOI (e.g. "10.1109/TWC.2024.001"), a Semantic
            Scholar paperId, or a direct publisher URL.
        force_refresh: If True, skip the cache and re-fetch from the
            publisher.

    Returns:
        A YAML-structured result with fields:
          - status: ok | cache_hit | fallback | error
          - cache_hit: true | false
          - source: IEEE | Springer | ScienceDirect | Other | fallback
          - url: the resolved publisher URL
          - fetched_at: ISO 8601 timestamp
          - html: raw HTML (present when status is ok or fallback)
          - fallback_reason: present when status is fallback
          - error: present when status is error
    """
    # orchestrate_fetch already returns a pre-serialized YAML string rather
    # than a dict, so it can't be shaped before the envelope the way other
    # tools' payloads are. Routing it through output.respond() still
    # guarantees the hard_cap bound (raw publisher HTML has no size limit
    # otherwise).
    return output.respond(publisher.orchestrate_fetch(identifier, force_refresh=force_refresh))
