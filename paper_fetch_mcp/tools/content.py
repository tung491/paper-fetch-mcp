"""Content retrieval tools: get_content_by_citation_key, doi2content,
get_image_path, get_paper, list_figures.

Thin `@mcp.tool` wrappers over `core.papers` (design spec
`docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md`, §5,
§6, §8 "Phase 2", finding F8). This REPLACES the previous dead-fork
`tools/content.py`, which read `paper_fetch_mcp.db` directly and was never
wired into the new `core/*` stack; the old `_get_paper_content`/
`_get_content_by_doi` bodies now live, fixed (finding F5's fuzzy-resolve
bug), in `core/papers.py` as `get_paper_content`/`get_content_by_doi`.

`get_image_path` has no equivalent `core/*` helper (none of `corpus.py`,
`papers.py`, etc. own figure-path resolution) -- its small filesystem-glob
routine stays local here, built on `core.papers.resolve_citation_to_id` for
the one piece of business logic (citation-key -> paper id) it actually
needs. This mirrors the old tool's behaviour.

Phase 2 (finding F8, "a month of real usage"):

* ``doi2content`` normalises the DOI (``core.papers.normalize_doi``) and,
  on a miss, returns a structured ``{found: false, suggestions, next}``
  envelope instead of a bare "not found" string -- see
  ``core.papers.not_found_envelope``.
* ``get_content_by_citation_key`` does the same per-key on a miss. The hit
  shape for both tools is UNCHANGED (still the same content string / dict
  of citation_key -> content string) -- only the miss shape changed, and
  neither tool's input schema changed, so both stay byte-identical in the
  golden tool contract (`tests/golden/tools_list.json`).
* ``get_paper`` is the new umbrella tool (spec §6): resolves a citation
  key, DOI, arXiv id, or title via ``core.papers.resolve_any_identifier``
  and returns one consistent envelope regardless of identifier kind, with
  ``max_chars_per_paper``/``offset`` paging over the assembled content.
  ``doi2content``/``get_content_by_citation_key`` are conceptually aliases
  onto the same resolution logic per spec §6, but keep their own existing
  signatures and hit-path output shapes -- see module note above. Its
  ``save_path`` arg writes the paper's FULL content to a file (local hits
  and online fetches alike) and reports the write in a ``saved`` block --
  the only way to get a whole long paper out of this tool in one call,
  since the result itself is windowed by ``max_chars_per_paper``.
* ``list_figures`` enumerates a paper's figure files so agents can
  discover real figure ids instead of guessing them (`get_image_path` was
  called 0 times in a month of transcripts despite 17 skill references).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from paper_fetch_mcp import mcp
from paper_fetch_mcp.config import MISTRAL_OUTPUT_DIR
from paper_fetch_mcp.core import online
from paper_fetch_mcp.core import output
from paper_fetch_mcp.core import papers

logger = logging.getLogger(__name__)


@mcp.tool(timeout=45)
def doi2content(doi: str) -> str:
    """
    Retrieve full paper content by DOI.

    The DOI is normalised before lookup (strips a "https://doi.org/",
    "http://dx.doi.org/", or "doi:" prefix and surrounding whitespace, then
    lowercases -- DOIs are case-insensitive), so
    "https://doi.org/10.1109/X" and "10.1109/x" resolve to the same paper.

    Args:
        doi: The DOI of the paper (e.g., "10.1109/...").

    Returns:
        YAML with the full markdown content of the paper if found. On a
        miss: a structured `{found: false, query, suggestions, next}`
        envelope -- `suggestions` are the nearest citation keys (fuzzy +
        title match) and `next` lists concrete follow-up tool calls (e.g.
        `ingest_arxiv <id>`, `fetch_publisher_html <doi>`) instead of a
        bare "not found" string.
    """
    if papers.resolve_doi_to_id(doi) is None:
        return output.respond(papers.not_found_envelope(doi))
    return output.respond(papers.get_content_by_doi(doi))


@mcp.tool(timeout=45)
def get_content_by_citation_key(
    citation_keys: List[str], sections: List[str] = ["*"]
) -> str:
    """
    Retrieve content for specific papers by citation key.

    Args:
        citation_keys: List of citation keys (e.g., ["Author2023Title"]).
                       Maximum 20 keys per request.
        sections: List of sections to retrieve. Options:
                  - "*": All available text chunks.
                  - "abstract": Abstract from database.
                  - "Introduction", "Experiment", "Conclusion",
                    "Proposed_System": specific chunks.

    Returns:
        YAML mapping each citation_key to its retrieved content string when
        found. A key that misses maps instead to a structured
        `{found: false, query, suggestions, next}` envelope (nearest
        citation-key suggestions plus concrete follow-up actions) rather
        than a bare "not found" string.
    """
    citation_keys = citation_keys[:20]
    results: Dict[str, object] = {}
    for key in citation_keys:
        if papers.resolve_citation_to_id(key) is None:
            results[key] = papers.not_found_envelope(key)
        else:
            results[key] = papers.get_paper_content(key, sections)
    return output.respond(results)


# 90 s rather than the usual 45: on a corpus miss this tool goes to the
# network (doi.org redirect + publisher page, each with a 15-20 s timeout)
# then a headless-browser render on this host, and may finally retry
# through a relayed Tailscale exit node. Local hits
# return in milliseconds as before.
@mcp.tool(timeout=90)
def get_paper(
    identifier: str,
    max_chars_per_paper: int = 12000,
    offset: int = 0,
    save_path: str = "",
) -> str:
    """
    Umbrella paper lookup (spec §6): resolves a citation key, DOI, arXiv
    id, or title to the same paper, regardless of which kind of identifier
    was given. `doi2content` and `get_content_by_citation_key` are
    conceptually aliases onto this same resolution logic, but keep their
    own existing signatures and output shapes (see this module's
    docstring) -- use `get_paper` directly when the identifier's kind is
    unknown or when paging through a long paper.

    If the paper is NOT in the local corpus, it is fetched online instead:
    the publisher page for a DOI/URL, arxiv.org for an arXiv id, or a
    Semantic Scholar title lookup first for a bare title. If this host's
    own network cannot access the paper (paywall 403, geo-block,
    connection refused/timed out, or a 200 that is only a paywall stub),
    the fetch is retried once through the Tailscale exit node `ubuntu-lab`
    (lab network with institutional access). The envelope then carries
    `source: online` and `route: direct | exit_node | cache` so you can
    tell how it was obtained; `found: false` misses carry an `online`
    block saying what was tried and whether the exit node was available.

    Args:
        identifier: A citation key, DOI (any form -- normalised), arXiv id
            (bare, "arXiv:"-prefixed, or a full arxiv.org URL), or paper
            title.
        max_chars_per_paper: Maximum content characters to return in one
            call (default 12000).
        offset: Character offset into the assembled content to start the
            returned window at (default 0) -- page through a long paper
            with repeated calls, e.g. `offset=12000`, `offset=24000`, ...
        save_path: Optional file (or directory) to write the paper's FULL
            content to -- the whole assembled paper, not just the window
            this call returns, so one call with `save_path` gets a long
            paper onto disk without paging it through the tool result.
            Works for an online hit too (what was fetched is written).
            A relative path is resolved under `PAPER_SAVE_DIR`
            (default `<repo>/saved_papers`), an absolute path is used
            as given; a path that is an existing directory, ends with
            "/", or has no extension is treated as a directory and the
            file inside it is named `<citation_key>.md`. Missing parents
            are created; an existing file is overwritten (`overwrote:
            true` in the report).

    Returns:
        YAML `{found: true, citation_key, id, doi, title, content,
        paging: {offset, returned_chars, total_chars, truncated}}` on a
        local hit; `{found: true, source: online, kind, doi, arxiv_id, url,
        publisher, status, route, content, paging, ...}` on an online hit;
        or `{found: false, query, suggestions, next, online: {...}}` when
        neither the corpus nor the network (direct or via exit node) had
        it. With `save_path` set, a hit also carries `saved: {path, chars,
        bytes, overwrote}` -- or `saved: {path, error}` if the write
        failed, which does not fail the lookup itself.
    """
    result = papers.get_paper(
        identifier,
        max_chars_per_paper=max_chars_per_paper,
        offset=offset,
        save_path=save_path,
    )
    if not result.get("found"):
        result = online.get_paper_online(
            identifier,
            local_miss=result,
            max_chars_per_paper=max_chars_per_paper,
            offset=offset,
            save_path=save_path,
        )
    return output.respond(result)


@mcp.tool(timeout=45)
def list_figures(citation_key: str) -> str:
    """
    Enumerate the figure files available for a paper, so an agent can
    discover real figure ids instead of guessing them before calling
    `get_image_path` (finding F8: `get_image_path` was called 0 times in a
    month of transcripts despite 17 skill references -- agents had no way
    to learn what figure ids exist for a given paper).

    Args:
        citation_key: The paper's citation key (or its directory id).

    Returns:
        YAML `{found: true, citation_key, id, figures: [{figure_id, path}],
        count}` on a hit, or `{found: false, query, suggestions, next}` on
        a miss.
    """
    return output.respond(papers.list_figures(citation_key), list_key="figures")


@mcp.tool(timeout=45)
def get_image_path(citation_key: str, figure_id: str) -> str:
    """
    Get the absolute file path for a specific figure in a paper.

    Args:
        citation_key: The paper's citation key.
        figure_id: The figure filename (e.g., "chunk_0_img-0.jpg").

    Returns:
        YAML with the absolute path or an error message.
    """
    paper_data = papers.resolve_citation_to_id(citation_key)
    if not paper_data:
        return output.error("Paper not found.")

    paper_id = paper_data["id"]

    possible_paths = [
        MISTRAL_OUTPUT_DIR / paper_id / "figures" / figure_id,
        MISTRAL_OUTPUT_DIR / paper_id / figure_id,
        MISTRAL_OUTPUT_DIR / paper_id / "figures" / f"{figure_id}.png",
        MISTRAL_OUTPUT_DIR / paper_id / "figures" / f"{figure_id}.jpg",
    ]

    if paper_data.get("file"):
        stem = Path(paper_data["file"]).stem
        possible_paths.extend(
            [
                MISTRAL_OUTPUT_DIR / stem / "figures" / figure_id,
                MISTRAL_OUTPUT_DIR / stem / "figures" / f"{figure_id}.png",
                MISTRAL_OUTPUT_DIR / stem / figure_id,
            ]
        )

    for p in possible_paths:
        if p.exists():
            return output.respond({"path": str(p)})

    return output.error(f"Image not found. Tried {len(possible_paths)} paths.")
