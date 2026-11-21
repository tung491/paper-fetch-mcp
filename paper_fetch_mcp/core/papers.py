"""papers.db access and paper-content assembly helpers.

Merges the former ``paper_fetch_mcp/db.py`` (all SQLite access against
``papers.db``) with the content-assembly helpers from
``paper_fetch_mcp/tools/content.py`` (lines 15-107). See design spec
docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md §5,
findings F5 and F8.

Deliberate decisions carried over from the two diverged copies (see the
Wave 2b task brief for the full rationale):

* ``_fuzzy_resolve`` previously did ``SELECT id, file, abstract FROM
  papers`` and then read ``row["citation_key"]`` -- an unselected column,
  which raises ``IndexError`` on the first row and was swallowed by a
  broad ``except Exception``, so fuzzy citation-key resolution silently
  returned ``None`` for every non-empty table. Fixed here by adding
  ``citation_key`` to that SELECT's projection and to the two LIKE
  fallback SELECTs.
* Abstract-inclusion semantics for ``get_paper_content`` follow
  ``tools/content.py`` (``"abstract" in sections or "*" in sections``),
  not ``mcp_server_v2.py``'s narrower ``"abstract" in sections``. This is
  a deliberate, live behaviour change: ``hyde_query``/``stepback_query``
  (which go through the ``mcp_server_v2.py`` copy today) will gain the DB
  abstract on a bare ``"*"`` once a later wave cuts them over to this
  module -- flag it in review rather than treating it as a mystery diff.
* ``cached_html`` / ``_ensure_cached_html_table`` are NOT here. That is
  publisher-cache state owned by a concurrent agent moving it to
  ``core/publisher.py``.

No side effects at import time: no DB connection, no ``CREATE TABLE``,
nothing runs until a function in this module is called. ``init_fts()`` is
invoked from the FastMCP lifespan, never here at import.
"""

import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from paper_fetch_mcp.config import DB_PATH, MISTRAL_OUTPUT_DIR, PAPER_SAVE_DIR
from paper_fetch_mcp.core import images

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Connection management
# --------------------------------------------------------------------------


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Citation / DOI resolution
# --------------------------------------------------------------------------


def _normalize_key(key: str) -> str:
    """Normalize a citation key for fuzzy comparison:
    lowercase, strip whitespace, deduplicate same-year repeats.

    e.g. "ziyuMilcom20252025" -> "ziyubilcommilcom2025"
    Uses backreference \\1 to match only a literal repeat of the same 4 digits.
    """
    key = key.strip().lower()
    key = re.sub(r"\s+", "", key)  # remove whitespace
    key = re.sub(r"(\d{4})\1", r"\1", key)  # deduplicate "YYYYYYYY" -> "YYYY"
    return key


def _fuzzy_resolve(citation_key: str) -> Optional[dict]:
    """Fuzzy resolution: try normalized match + prefix/LIKE fallback.

    Strategy:
      1. Exact citation_key match  (already tried by caller)
      2. Exact id match            (already tried by caller)
      3. Normalized citation_key match (handles duplicated years, case, whitespace)
      4. Normalized id match
      5. LIKE %prefix% on citation_key (handles truncated/abbreviated keys)
      6. LIKE %prefix% on id
      7. If exactly one row matches any of the above, return it.

    NOTE: the projections below MUST include ``citation_key`` -- this is
    the fix for finding F5. The original code selected only
    ``id, file, abstract`` and then read ``row["citation_key"]`` below,
    which raises ``IndexError`` (an unselected column) on the very first
    row, silently caught by the broad ``except Exception`` at the bottom
    of this function, so fuzzy resolution always returned ``None``. See
    ``tests/test_core_papers.py::test_fuzzy_resolve_regression_missing_column``
    for a test that fails against the old buggy SELECT.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            norm = _normalize_key(citation_key)

            # Fetch all rows for fuzzy comparison.
            cursor.execute("SELECT id, file, abstract, citation_key FROM papers")
            rows = cursor.fetchall()

            candidates = []
            for row in rows:
                row_ck = (row["citation_key"] or "").strip()
                row_id = (row["id"] or "").strip()

                # Strategy 3: normalized match
                if _normalize_key(row_ck) == norm or _normalize_key(row_id) == norm:
                    candidates.append(row)

            # Strategy 5/6: prefix-based LIKE -- user-supplied key is a prefix of
            # stored key, e.g. "ziyuMilcom2025" is prefix of "ziyuMilcom20252025"
            if not candidates:
                like = f"{norm}%"
                cursor.execute(
                    "SELECT id, file, abstract, citation_key FROM papers "
                    "WHERE citation_key LIKE ?",
                    (like,),
                )
                candidates = cursor.fetchall()
                if not candidates:
                    cursor.execute(
                        "SELECT id, file, abstract, citation_key FROM papers "
                        "WHERE id LIKE ?",
                        (like,),
                    )
                    candidates = cursor.fetchall()

            if len(candidates) == 1:
                return dict(candidates[0])
            elif len(candidates) > 1:
                logger.warning(
                    "Ambiguous citation key '%s' matched %d rows; returning first.",
                    citation_key, len(candidates),
                )
                return dict(candidates[0])

            return None
    except Exception as e:
        logger.error("Fuzzy DB lookup failed for key '%s': %s", citation_key, e)
        return None


def resolve_citation_to_id(citation_key: str) -> Optional[dict]:
    """Resolve a citation key to paper details using papers.db.

    Resolution order:
      1. Exact citation_key match
      2. Exact id match (directory/file name)
      3. Normalized match (handles duplicated years, whitespace, case)
      4. Prefix LIKE match (handles truncated keys)
    """
    citation_key = citation_key.strip()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 1. Exact citation_key
            cursor.execute(
                "SELECT id, file, abstract FROM papers WHERE citation_key = ?",
                (citation_key,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            # 2. Exact id
            cursor.execute(
                "SELECT id, file, abstract FROM papers WHERE id = ?",
                (citation_key,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        # 3 & 4: fuzzy fallback (needs full table scan, outside the with block to
        # avoid holding the connection open during the scan)
        return _fuzzy_resolve(citation_key)
    except Exception as e:
        logger.error("DB lookup failed for key '%s': %s", citation_key, e)
        return None


def normalize_doi(doi: str) -> str:
    """Normalize a DOI for lookup (finding F8: ``doi2content`` missed 88% of
    the time -- in part because ``https://doi.org/10.1109/X`` and
    ``10.1109/x`` were treated as different strings).

    Strips a leading ``https://doi.org/``, ``http://dx.doi.org/`` or ``doi:``
    prefix (case-insensitively), surrounding whitespace, then lowercases the
    result (DOIs are case-insensitive per the DOI Handbook).
    """
    if not doi:
        return doi
    doi = doi.strip()
    doi = _DOI_PREFIX_RE.sub("", doi)
    return doi.strip().lower()


_DOI_PREFIX_RE = re.compile(r"^\s*(?:https?://(?:dx\.|www\.)?doi\.org/|doi:\s*)", re.IGNORECASE)

# Bare arXiv id, e.g. "2301.12345" or "2301.12345v2" (also matches when
# embedded in "arXiv:2301.12345" or a full arxiv.org URL).
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

# A DOI has the form "10.<4-9 digit registrant>/<suffix>" (DOI Handbook
# §2). Used only to decide which "next" follow-up action to suggest on a
# miss -- not for lookup (normalize_doi handles that).
_DOI_SHAPE_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def _looks_like_doi(identifier: str) -> bool:
    return bool(_DOI_SHAPE_RE.match(normalize_doi(identifier)))


def _extract_arxiv_id(identifier: str) -> Optional[str]:
    m = _ARXIV_ID_RE.search(identifier or "")
    return m.group(1) if m else None


def looks_like_doi(identifier: str) -> bool:
    """Public form of ``_looks_like_doi`` (used by ``core.online`` to decide
    how to fetch a `get_paper` miss from the network)."""
    return _looks_like_doi(identifier)


def extract_arxiv_id(identifier: str) -> Optional[str]:
    """Public form of ``_extract_arxiv_id``: the bare ``YYMM.NNNNN`` id found
    in ``identifier`` (an "arXiv:" prefix, arxiv.org URL, or version suffix
    is tolerated), or ``None``."""
    return _extract_arxiv_id(identifier)


def resolve_doi_to_id(doi: str) -> Optional[dict]:
    """Resolve a DOI to paper details using papers.db.

    The lookup DOI and the stored ``doi`` column are both normalized
    (``normalize_doi``) before comparison, so
    ``https://doi.org/10.1109/X``, ``doi:10.1109/x`` and ``10.1109/X`` (and
    any DOI stored in the DB with different casing) all resolve to the same
    row.
    """
    normalized = normalize_doi(doi)
    if not normalized:
        return None
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, file, abstract, citation_key FROM papers "
                "WHERE lower(doi) = ?",
                (normalized,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error("DB lookup by DOI failed: %s", e)
        return None


def _full_row_by_id(paper_id: str) -> Optional[dict]:
    """Fetch the full projection (id/title/doi/citation_key/abstract/file/year)
    for a known paper id -- used to give ``get_paper``/``list_figures``/
    ``suggest_citation_keys`` a consistent row shape regardless of which
    resolution path (exact, fuzzy, DOI, FTS) found the paper."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, doi, citation_key, abstract, file, year "
                "FROM papers WHERE id = ?",
                (paper_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error("Full row lookup failed for id '%s': %s", paper_id, e)
        return None


def resolve_any_identifier(identifier: str) -> Optional[dict]:
    """Resolve ``identifier`` as a citation key, DOI, arXiv id, or title --
    the shared resolution ladder behind ``get_paper`` (spec §6 umbrella
    tool). Returns the full row projection (see ``_full_row_by_id``) or
    ``None``.

    Order: 1) citation key / directory id (exact, then fuzzy, via
    ``resolve_citation_to_id``) 2) DOI (any of the normalised forms)
    3) a bare arXiv id extracted from the string (handles "arXiv:...", a
    full arxiv.org URL, or a version suffix) 4) title, via the FTS/LIKE
    search -- only accepted when it resolves unambiguously (a single hit,
    or a single case-insensitive exact title match among several).
    """
    identifier = (identifier or "").strip()
    if not identifier:
        return None

    hit = resolve_citation_to_id(identifier)
    if hit:
        return _full_row_by_id(hit["id"])

    hit = resolve_doi_to_id(identifier)
    if hit:
        return _full_row_by_id(hit["id"])

    arxiv_id = _extract_arxiv_id(identifier)
    if arxiv_id and arxiv_id != identifier:
        hit = resolve_citation_to_id(arxiv_id)
        if hit:
            return _full_row_by_id(hit["id"])

    candidates = search_papers_fts(identifier, limit=5)
    if len(candidates) == 1:
        return _full_row_by_id(candidates[0]["id"])
    if candidates:
        exact = [
            c
            for c in candidates
            if (c.get("title") or "").strip().lower() == identifier.lower()
        ]
        if len(exact) == 1:
            return _full_row_by_id(exact[0]["id"])

    return None


def suggest_citation_keys(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Nearest citation-key suggestions for a miss (finding F8): the fixed
    fuzzy resolver (``_fuzzy_resolve``, F5) plus an FTS/title match,
    deduped by paper id.

    Returns plain dicts with ``id``/``citation_key``/``title`` -- no DB
    connection is exposed past this function.
    """
    query = (query or "").strip()
    if not query:
        return []

    seen_ids: set = set()
    suggestions: List[Dict[str, Any]] = []

    def _add(row: Optional[dict]) -> None:
        if not row:
            return
        pid = row.get("id")
        if not pid or pid in seen_ids:
            return
        seen_ids.add(pid)
        suggestions.append(
            {
                "id": pid,
                "citation_key": row.get("citation_key"),
                "title": row.get("title"),
            }
        )

    fuzzy_hit = _fuzzy_resolve(query)
    if fuzzy_hit:
        _add(_full_row_by_id(fuzzy_hit["id"]))

    for row in search_papers_fts(query, limit=limit):
        if len(suggestions) >= limit:
            break
        _add(row)

    return suggestions[:limit]


def not_found_envelope(query: str, *, limit: int = 5) -> Dict[str, Any]:
    """Build the structured ``{found: false, ...}`` miss envelope (finding
    F8): nearest ``suggestions`` plus concrete ``next`` follow-up actions,
    so a miss gives the agent somewhere to go instead of a bare string.
    """
    suggestions = suggest_citation_keys(query, limit=limit)

    next_actions: List[str] = []
    arxiv_id = _extract_arxiv_id(query)
    if arxiv_id:
        next_actions.append(f"ingest_arxiv {arxiv_id}")
    if _looks_like_doi(query):
        next_actions.append(f"fetch_publisher_html {query}")
    if not next_actions:
        next_actions = ["search_metadata", "semantic_scholar_search"]

    return {
        "found": False,
        "query": query,
        "suggestions": suggestions,
        "next": next_actions,
    }


# --------------------------------------------------------------------------
# Paper content assembly (merged from tools/content.py:15-107)
# --------------------------------------------------------------------------


def get_paper_content(citation_key: str, sections: List[str]) -> str:
    """Retrieve content for a single paper by citation key.

    Args:
        citation_key: The paper's citation key.
        sections: List of sections to retrieve. "*" for all, "abstract" for
                  abstract, or specific section names like "Introduction".

    Abstract-inclusion semantics: the abstract is included when
    ``"abstract"`` OR ``"*"`` is in ``sections`` (tools/content.py's
    semantics, deliberately adopted over mcp_server_v2.py's narrower
    ``"abstract" in sections`` check -- see module docstring).
    """
    paper_data = resolve_citation_to_id(citation_key)
    if not paper_data:
        return f"Paper not found for key: '{citation_key}'"

    paper_id = paper_data["id"]
    paper_file_stem = (
        Path(paper_data["file"]).stem if paper_data.get("file") else paper_id
    )

    # Resolve chunk directory
    chunk_dir = MISTRAL_OUTPUT_DIR / paper_id / "chunk"
    if not chunk_dir.exists():
        chunk_dir = MISTRAL_OUTPUT_DIR / paper_file_stem / "chunk"

    content_parts = []

    # Handle abstract request
    if "abstract" in sections or "*" in sections:
        abstract_text = paper_data.get("abstract", "")
        if abstract_text:
            content_parts.append(f"# Abstract\n{abstract_text}")

    # Handle file-based sections
    if chunk_dir.exists():
        available_files = sorted(chunk_dir.glob("*.md"))

        for f in available_files:
            include = False
            if "*" in sections:
                include = True
            else:
                fname = f.name.lower()
                for sec_type in sections:
                    if sec_type.lower() == "abstract":
                        continue  # Handled above
                    if sec_type.lower() in fname:
                        include = True
                        break

            if include:
                try:
                    text = f.read_text(encoding="utf-8")
                    content_parts.append(text)
                except Exception as e:
                    logger.error("Error reading %s: %s", f, e)

    if not content_parts:
        return "No content found for requested sections."
    return "\n\n".join(content_parts)


def get_content_by_doi(doi: str) -> str:
    """Retrieve full content by DOI."""
    paper_data = resolve_doi_to_id(doi)
    if not paper_data:
        return f"Paper with DOI '{doi}' not found in database."

    if paper_data.get("citation_key"):
        return get_paper_content(paper_data["citation_key"], sections=["*"])

    paper_id = paper_data["id"]
    paper_file_stem = (
        Path(paper_data["file"]).stem if paper_data.get("file") else paper_id
    )

    chunk_dir = MISTRAL_OUTPUT_DIR / paper_id / "chunk"
    if not chunk_dir.exists():
        chunk_dir = MISTRAL_OUTPUT_DIR / paper_file_stem / "chunk"

    content_parts = []

    abstract_text = paper_data.get("abstract", "")
    if abstract_text:
        content_parts.append(f"# Abstract\n{abstract_text}")

    if chunk_dir.exists():
        for f in sorted(chunk_dir.glob("*.md")):
            try:
                content_parts.append(f.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error("Error reading %s: %s", f, e)

    if not content_parts:
        return "No content found for this DOI."
    return "\n\n".join(content_parts)


# --------------------------------------------------------------------------
# get_paper: new umbrella tool (spec §6, Phase 2 deliverable 3)
# --------------------------------------------------------------------------


def page_window(
    full_content: str, offset: int, max_chars: int
) -> Tuple[str, Dict[str, Any]]:
    """Slice ``full_content`` into the ``[offset, offset + max_chars)`` window
    and describe it with the ``paging`` block `get_paper` returns
    (``{offset, returned_chars, total_chars, truncated}``). Shared with
    ``core.online`` so a paper fetched from the network pages identically
    to one served from the corpus.
    """
    total_chars = len(full_content)
    offset = max(0, offset)
    max_chars = max(0, max_chars)
    window = full_content[offset : offset + max_chars]
    paging = {
        "offset": offset,
        "returned_chars": len(window),
        "total_chars": total_chars,
        "truncated": (offset + len(window)) < total_chars,
    }
    return window, paging


def save_stem(*candidates: Optional[str]) -> str:
    """Filesystem-safe stem for a saved paper file, from the first non-empty
    candidate (citation key, arXiv id, DOI, title, raw identifier)."""
    for cand in candidates:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (cand or "").strip()).strip("._-")
        if slug:
            return slug[:120]
    return "paper"


def save_paper_content(
    full_content: str,
    save_path: str,
    default_stem: str = "paper",
    local_images: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    """Write ``full_content`` to ``save_path`` and describe the write for the
    ``saved`` block `get_paper` returns.

    Path handling:

    * ``~`` and ``$VARs`` are expanded; a RELATIVE path is resolved under
      ``PAPER_SAVE_DIR`` (config), not the server process's cwd -- the MCP
      server's cwd is whatever launched it, so a bare ``"smith2024.md"``
      would otherwise land somewhere the caller cannot predict.
    * A path that already exists as a directory, ends with a separator, or
      has no file extension is treated as a DIRECTORY; the file inside it is
      named ``<default_stem>.md``.
    * Missing parent directories are created.
    * Images the content links to are copied into ``<file stem>_files/``
      beside the file and the links rewritten to point there
      (``core.images.localize``): remote figures from the figure cache or a
      download, relative corpus figures (``chunk_0_img-0.jpg``) from
      ``local_images`` (file name -> path). The report then carries
      ``images: {dir, saved, failed}``; an image that cannot be had keeps
      its original link.

    Returns ``{path, chars, bytes, overwrote}`` on success, or
    ``{path, error}`` on failure -- this never raises, so a save problem
    (read-only dir, bad path) degrades the ``get_paper`` call to "content
    returned, not saved" instead of losing the fetched paper entirely.
    """
    raw = (save_path or "").strip()
    if not raw:
        return {"error": "save_path was empty"}

    try:
        target = Path(os.path.expandvars(raw)).expanduser()
        if not target.is_absolute():
            target = Path(PAPER_SAVE_DIR) / target
        # Resolved first, THEN classified: a relative "notes/" must be tested
        # for dir-ness under PAPER_SAVE_DIR, not against the process cwd.
        if raw.endswith(("/", os.sep)) or target.is_dir() or not target.suffix:
            target = target / f"{save_stem(default_stem)}.md"

        overwrote = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        data, image_report = images.localize(
            full_content or "", target, local_files=local_images
        )
        target.write_text(data, encoding="utf-8")
    except Exception as e:  # noqa: BLE001 -- reported, never raised to the tool
        logger.warning("save_paper_content(%r) failed: %s", save_path, e)
        return {"path": raw, "error": f"{type(e).__name__}: {e}"}

    logger.info("saved paper content -> %s (%d chars)", target, len(data))
    report: Dict[str, Any] = {
        "path": str(target),
        "chars": len(data),
        "bytes": len(data.encode("utf-8")),
        "overwrote": overwrote,
    }
    if image_report:
        report["images"] = image_report
    return report


def get_paper(
    identifier: str,
    max_chars_per_paper: int = 12000,
    offset: int = 0,
    save_path: str = "",
) -> Dict[str, Any]:
    """Resolve ``identifier`` (citation key, DOI, arXiv id, or title) via
    ``resolve_any_identifier`` and return one consistent envelope regardless
    of which kind of identifier was given.

    On a hit: ``{found: True, citation_key, id, doi, title, content,
    paging: {offset, returned_chars, total_chars, truncated}}``, where
    ``content`` is a window of the full assembled paper content
    (``get_paper_content`` with ``sections=["*"]``) starting at ``offset``
    and at most ``max_chars_per_paper`` chars long -- so a large paper can
    be paged through with repeated calls rather than always paying for (or
    being truncated by ``core.output``'s budget on) the whole thing.

    On a miss: ``not_found_envelope(identifier)`` -- ``{found: False,
    query, suggestions, next}``.

    With ``save_path`` set, the paper's FULL assembled content (not the
    returned window) is also written to disk via ``save_paper_content`` and
    the hit envelope carries its ``saved: {path, chars, bytes, overwrote}``
    report -- or ``saved: {path, error}`` if the write failed, which does
    NOT fail the lookup itself.

    NOTE: the paging key is deliberately NOT named ``meta`` -- `core.output
    .respond()`'s envelope wrapper unconditionally overwrites a top-level
    ``meta`` key on the payload it wraps, which would silently clobber this
    paging info.
    """
    identifier = (identifier or "").strip()
    resolved = resolve_any_identifier(identifier)
    if not resolved:
        return not_found_envelope(identifier)

    key = resolved.get("citation_key") or resolved["id"]
    full_content = get_paper_content(key, ["*"])
    window, paging = page_window(full_content, offset, max_chars_per_paper)

    envelope: Dict[str, Any] = {
        "found": True,
        "citation_key": resolved.get("citation_key"),
        "id": resolved.get("id"),
        "doi": resolved.get("doi"),
        "title": resolved.get("title"),
        "content": window,
        "paging": paging,
    }
    if (save_path or "").strip():
        # The FULL assembled content, not the returned window -- the point of
        # save_path is to get the whole paper onto disk in one call instead of
        # paging it through the tool result.
        envelope["saved"] = save_paper_content(
            full_content,
            save_path,
            default_stem=save_stem(
                resolved.get("citation_key"), resolved.get("id"), identifier
            ),
            local_images={
                fig["figure_id"]: Path(fig["path"])
                for fig in list_figures(key).get("figures", [])
            },
        )
    return envelope


# --------------------------------------------------------------------------
# list_figures: new tool (spec §6, Phase 2 deliverable 4)
# --------------------------------------------------------------------------


def list_figures(citation_key: str) -> Dict[str, Any]:
    """Enumerate ``mistral_output/<id>/*.jpg|jpeg|png`` for a paper so
    agents can discover real figure ids instead of guessing them (finding
    F8: ``get_image_path`` was called 0 times in a month because agents had
    no way to learn what figure ids exist).

    Returns ``{found: True, citation_key, id, figures: [{figure_id, path}],
    count}`` on a hit, or ``not_found_envelope(citation_key)`` on a miss.
    """
    paper_data = resolve_citation_to_id(citation_key)
    if not paper_data:
        return not_found_envelope(citation_key)

    # resolve_citation_to_id's exact-match branches only project
    # id/file/abstract (not citation_key) -- fetch the full row so the
    # returned envelope's citation_key is always populated, even when the
    # caller passed the bare directory id rather than the citation key.
    full_row = _full_row_by_id(paper_data["id"]) or paper_data

    paper_id = paper_data["id"]
    paper_file_stem = (
        Path(paper_data["file"]).stem if paper_data.get("file") else paper_id
    )

    # Real corpus layout has figures directly under MISTRAL_OUTPUT_DIR/<id>/
    # (e.g. "chunk_0_img-0.jpg"), not under a "figures" subdirectory -- but
    # `get_image_path` also checks a "figures" subdir, so both are scanned
    # here for parity with that tool's own path list.
    candidate_dirs = []
    for stem in dict.fromkeys([paper_id, paper_file_stem]):  # de-dup, keep order
        candidate_dirs.append(MISTRAL_OUTPUT_DIR / stem)
        candidate_dirs.append(MISTRAL_OUTPUT_DIR / stem / "figures")

    figures: List[Dict[str, str]] = []
    seen_names: set = set()
    for d in candidate_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for p in sorted(d.glob(pattern)):
                if p.name in seen_names:
                    continue
                seen_names.add(p.name)
                figures.append({"figure_id": p.name, "path": str(p)})

    return {
        "found": True,
        "citation_key": full_row.get("citation_key") or citation_key,
        "id": paper_id,
        "figures": figures,
        "count": len(figures),
    }


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def search_papers_fts(query: str, limit: int = 20) -> List[dict]:
    """Full-text search across title, abstract, and author fields.

    Requires the FTS5 virtual table to be initialized via ``init_fts()``.
    Falls back to LIKE-based search if the FTS table doesn't exist (it is
    not initialised on the live DB today: only the unused package
    entrypoint ever called ``init_fts()``).
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Try FTS first
            try:
                cursor.execute(
                    "SELECT id, title, author, year, abstract, citation_key, doi "
                    "FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?",
                    (query, limit),
                )
                rows = cursor.fetchall()
                if rows:
                    return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                pass  # FTS table doesn't exist, fall back

            # Fallback: LIKE search
            like_q = f"%{query}%"
            cursor.execute(
                "SELECT id, title, author, year, abstract, citation_key, doi "
                "FROM papers WHERE title LIKE ? OR abstract LIKE ? OR author LIKE ? "
                "LIMIT ?",
                (like_q, like_q, like_q, limit),
            )
            return [dict(r) for r in cursor.fetchall()]
    except Exception as e:
        logger.error("FTS search failed: %s", e)
        return []


def search_by_author(author_name: str, limit: int = 50) -> List[dict]:
    """Search papers by author name (substring match)."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, author, year, citation_key, doi "
                "FROM papers WHERE author LIKE ? LIMIT ?",
                (f"%{author_name}%", limit),
            )
            return [dict(r) for r in cursor.fetchall()]
    except Exception as e:
        logger.error("Author search failed: %s", e)
        return []


def search_by_year_range(start_year: int, end_year: int, limit: int = 100) -> List[dict]:
    """Search papers by publication year range."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, author, year, citation_key, doi "
                "FROM papers WHERE year >= ? AND year <= ? ORDER BY year DESC LIMIT ?",
                (start_year, end_year, limit),
            )
            return [dict(r) for r in cursor.fetchall()]
    except Exception as e:
        logger.error("Year range search failed: %s", e)
        return []


def init_fts() -> None:
    """Initialize FTS5 virtual table for full-text search (idempotent).

    Called from the FastMCP lifespan -- never at import time.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5("
                "id, title, author, year, abstract, citation_key, doi, "
                "content='papers', content_rowid='rowid'"
                ")"
            )
            # Rebuild index from papers table
            cursor.execute(
                "INSERT OR REPLACE INTO papers_fts(papers_fts) VALUES('rebuild')"
            )
            conn.commit()
            logger.info("FTS5 index initialized/rebuilt")
    except Exception as e:
        logger.warning("FTS5 init failed (non-fatal): %s", e)


def search_metadata(
    query: Optional[str] = None,
    author: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Unified metadata search, replacing tools/utils.py's ``search_papers``,
    ``search_papers_by_author`` and ``search_papers_by_year`` (spec §6).

    Dispatches on which of ``query``/``author``/``year_from``/``year_to``
    are given:

    * Exactly one filter -> delegates to the matching single-purpose
      helper above, so behaviour (including FTS-then-LIKE degradation for
      ``query``) is identical to the tool it replaces.
    * Two or more filters -> combined with AND via one LIKE-based SQL
      query. (FTS5 MATCH does not portably AND with a plain year-range
      predicate in one query, so the combined path always uses LIKE --
      this is a deliberate, documented tradeoff: a combined query+author
      or query+year search does not use the FTS index even if it exists.)
    * No filters -> returns an error dict rather than the whole table.

    Returns a plain dict; a later wave wraps this in the response
    envelope (this module does not import ``output``).
    """
    has_query = bool(query)
    has_author = bool(author)
    has_year = year_from is not None or year_to is not None

    filters_used = sum([has_query, has_author, has_year])

    if filters_used == 0:
        return {
            "error": "At least one of query, author, year_from, year_to must be provided.",
            "results": [],
            "count": 0,
        }

    if filters_used == 1:
        if has_query:
            results = search_papers_fts(query, limit)
            return {"query": query, "results": results, "count": len(results)}
        if has_author:
            results = search_by_author(author, limit)
            return {"author": author, "results": results, "count": len(results)}
        # has_year only
        start = year_from if year_from is not None else 0
        end = year_to if year_to is not None else 9999
        results = search_by_year_range(start, end, limit)
        return {
            "year_range": f"{start}-{end}",
            "results": results,
            "count": len(results),
        }

    # Two or more filters given: combine with AND in a single LIKE-based query.
    clauses = []
    params: List[Any] = []
    if has_query:
        like_q = f"%{query}%"
        clauses.append("(title LIKE ? OR abstract LIKE ? OR author LIKE ?)")
        params.extend([like_q, like_q, like_q])
    if has_author:
        clauses.append("author LIKE ?")
        params.append(f"%{author}%")
    if year_from is not None:
        clauses.append("year >= ?")
        params.append(year_from)
    if year_to is not None:
        clauses.append("year <= ?")
        params.append(year_to)

    where_sql = " AND ".join(clauses)
    sql = (
        "SELECT id, title, author, year, citation_key, doi "
        f"FROM papers WHERE {where_sql} ORDER BY year DESC LIMIT ?"
    )
    params.append(limit)

    results: List[dict] = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            results = [dict(r) for r in cursor.fetchall()]
    except Exception as e:
        logger.error("Combined metadata search failed: %s", e)

    return {
        "query": query,
        "author": author,
        "year_from": year_from,
        "year_to": year_to,
        "results": results,
        "count": len(results),
    }
