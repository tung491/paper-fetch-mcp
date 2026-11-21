"""Semantic Scholar client: process-wide rate limiter + retry + TTL cache.

Extracted verbatim (Phase 1, wave 1b) from the Phase-0-hotfixed
`paper_fetch_mcp/mcp_server_v2.py` (`_S2_*` constants, `_s2_rate_limit_wait`,
`_s2_cache_key`, `_s2_cache_get`, `_s2_cache_put`, `_s2_get`, plus the search
logic that used to live in the `semantic_scholar_search` @mcp.tool body,
lines 957-1116 as of the extraction).

This is the Phase 0 fix for F4 (0.5): a single naive `requests.get` before
this had no retry/limiter/cache and produced 29 silent HTTP 429s and 50 hard
errors mined from a month of transcripts. The process-wide 1 req/s token
bucket, <=3 retries honouring Retry-After, the 1 h / 256-entry TTL cache, and
the thread-safety locks are preserved exactly.

`paper_fetch_mcp/tools/external.py` is the older, weaker fork (no limiter, no
cache) -- not used as the basis for this module.

Phase 2 (spec §6/§8): `search()` now projects each raw S2 row to a compact
shape (title, year, venue, doi, arxiv_id, citation_count, abstract<=400
chars) and cross-references it against the local corpus (`core.papers`),
adding `in_corpus: <citation_key>` or `in_corpus: null` -- this is what lets
an agent pivot straight from a search hit to `get_content_by_citation_key`
instead of burning another S2 call (the 2nd most common call bigram mined
from transcripts). `get_paper()` is the new `semantic_scholar_paper` tool's
backing call, sharing the same rate limiter/retry/cache path and the same
compact projection.

No side effects at import time: no HTTP calls happen until `search()`/
`get_paper()` (or the private helpers) are called. The corpus cross-
reference reads `papers.db` (read-only) lazily, once per `search()`/
`get_paper()` call -- never at import.
"""

import json
import os
import re
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

import requests

from paper_fetch_mcp.core import papers

_S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_S2_MIN_INTERVAL_S = 1.0  # process-wide token bucket: 1 request/second
_S2_PER_ATTEMPT_TIMEOUT_S = 20
_S2_MAX_ATTEMPTS = 4  # 1 try + up to 3 retries
_S2_CACHE_TTL_S = 3600.0  # 1 hour
_S2_CACHE_MAX_ENTRIES = 256

_s2_rate_lock = threading.Lock()
_s2_last_call_monotonic = 0.0

_s2_cache_lock = threading.Lock()
_s2_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _s2_rate_limit_wait() -> None:
    """Block the calling thread until >= _S2_MIN_INTERVAL_S has elapsed since
    the last Semantic Scholar request was issued anywhere in this process.

    Thread-safe: tools run in FastMCP's thread pool now (0.2), so multiple
    concurrent sessions can call this at once — the lock makes the
    check-then-set atomic across threads.
    """
    global _s2_last_call_monotonic
    with _s2_rate_lock:
        now = time.monotonic()
        wait_s = _S2_MIN_INTERVAL_S - (now - _s2_last_call_monotonic)
        if wait_s > 0:
            time.sleep(wait_s)
        _s2_last_call_monotonic = time.monotonic()


def _s2_cache_key(path: str, params: Dict[str, Any]) -> str:
    return json.dumps({"path": path, "params": params}, sort_keys=True, default=str)


def _s2_cache_get(key: str) -> Optional[Dict[str, Any]]:
    now = time.monotonic()
    with _s2_cache_lock:
        entry = _s2_cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if now - ts >= _S2_CACHE_TTL_S:
            _s2_cache.pop(key, None)
            return None
        return value


def _s2_cache_put(key: str, value: Dict[str, Any]) -> None:
    with _s2_cache_lock:
        if len(_s2_cache) >= _S2_CACHE_MAX_ENTRIES and key not in _s2_cache:
            oldest_key = min(_s2_cache, key=lambda k: _s2_cache[k][0])
            _s2_cache.pop(oldest_key, None)
        _s2_cache[key] = (time.monotonic(), value)


def _s2_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """GET a Semantic Scholar Graph API path.

    - process-wide 1 req/s limiter (thread-safe)
    - up to 3 retries on 429/5xx with exponential backoff, honouring
      Retry-After when present
    - 20 s per-attempt timeout
    - 1 h TTL cache keyed on the full request (path + params), capped at
      _S2_CACHE_MAX_ENTRIES entries

    Returns {"ok": True, "data": <json>} or {"ok": False, "error": <str>}.
    Never returns a raw non-200 body dressed up as success.
    """
    cache_key = _s2_cache_key(path, params)
    cached = _s2_cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"{_S2_BASE_URL}/{path.lstrip('/')}"
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    backoff_s = 1.0
    last_error = "unknown error"

    for attempt in range(_S2_MAX_ATTEMPTS):
        _s2_rate_limit_wait()
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=_S2_PER_ATTEMPT_TIMEOUT_S
            )
        except Exception as e:
            last_error = f"request failed: {e}"
            if attempt < _S2_MAX_ATTEMPTS - 1:
                time.sleep(backoff_s)
                backoff_s *= 2
                continue
            break

        if response.status_code == 200:
            try:
                result: Dict[str, Any] = {"ok": True, "data": response.json()}
            except Exception as e:
                result = {"ok": False, "error": f"invalid JSON response: {e}"}
            _s2_cache_put(cache_key, result)
            return result

        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if attempt < _S2_MAX_ATTEMPTS - 1:
                retry_after = response.headers.get("Retry-After")
                delay_s = backoff_s
                if retry_after:
                    try:
                        delay_s = max(delay_s, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(delay_s)
                backoff_s *= 2
                continue
            break

        # Non-retryable 4xx: fail immediately, do not cache.
        return {"ok": False, "error": f"HTTP {response.status_code}: {response.text[:300]}"}

    return {"ok": False, "error": last_error}


_S2_FIELDS = "paperId,title,authors,year,venue,abstract,externalIds,citationCount"

_MAX_ABSTRACT_CHARS = 400
_TRUNC_MARKER = "…"

# Corpus DOI values for arXiv-sourced papers are stored as
# "10.48550/arXiv.<id>" (see pre_processing_v2/papers.db). This regex pulls
# the bare arXiv id back out of a normalised (lowercased) DOI string so it
# can be compared against Semantic Scholar's externalIds.ArXiv. Kept as its
# own tiny regex here (rather than reusing core.papers' private
# `_extract_arxiv_id`/`_ARXIV_ID_RE`, which are for extracting an id out of
# an arbitrary user-supplied identifier/URL, not a DOI) per the task's "keep
# it simple" guidance.
_ARXIV_IN_DOI_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

_TITLE_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def _truncate_abstract(abstract: Optional[str]) -> Optional[str]:
    if not abstract:
        return abstract
    if len(abstract) <= _MAX_ABSTRACT_CHARS:
        return abstract
    return abstract[: _MAX_ABSTRACT_CHARS - len(_TRUNC_MARKER)].rstrip() + _TRUNC_MARKER


def _normalize_title(title: Optional[str]) -> str:
    """Cheap, deliberately simple title normalisation for the fallback
    corpus-match path: lowercase and drop everything that isn't a letter or
    digit, so punctuation/spacing/hyphenation differences between the S2
    title and the corpus title don't defeat an otherwise-exact match."""
    if not title:
        return ""
    return _TITLE_NONALNUM_RE.sub("", title.lower())


def _corpus_lookup() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Build (doi -> citation_key, arxiv_id -> citation_key, normalised
    title -> citation_key) maps from the local corpus in ONE query, so
    cross-referencing N search results costs one DB round trip, not N
    (the corpus is 1,648 papers).

    Reuses `core.papers.normalize_doi` (F8's DOI-normalisation fix) so a S2
    DOI in any of its accepted forms compares equal to whatever form the
    corpus happens to have stored. Never raises: a DB error just yields
    empty maps, so `in_corpus` degrades to `null` instead of failing the
    whole search.
    """
    doi_map: Dict[str, str] = {}
    arxiv_map: Dict[str, str] = {}
    title_map: Dict[str, str] = {}
    try:
        with papers.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doi, citation_key, title FROM papers")
            rows = cursor.fetchall()
    except Exception:
        return doi_map, arxiv_map, title_map

    for row in rows:
        citation_key = (row["citation_key"] or "").strip()
        if not citation_key:
            continue

        doi = row["doi"]
        if doi:
            norm_doi = papers.normalize_doi(doi)
            if norm_doi:
                doi_map.setdefault(norm_doi, citation_key)
                m = _ARXIV_IN_DOI_RE.search(norm_doi)
                if m:
                    arxiv_map.setdefault(m.group(1), citation_key)

        title = row["title"]
        if title:
            norm_title = _normalize_title(title)
            if norm_title:
                title_map.setdefault(norm_title, citation_key)

    return doi_map, arxiv_map, title_map


def _match_in_corpus(
    doi: Optional[str],
    arxiv_id: Optional[str],
    title: Optional[str],
    doi_map: Dict[str, str],
    arxiv_map: Dict[str, str],
    title_map: Dict[str, str],
) -> Optional[str]:
    """DOI first, then arXiv id, then normalised title -- per the task spec."""
    if doi:
        norm_doi = papers.normalize_doi(doi)
        if norm_doi and norm_doi in doi_map:
            return doi_map[norm_doi]

    if arxiv_id:
        m = _ARXIV_IN_DOI_RE.search(arxiv_id)
        norm_arxiv = m.group(1) if m else arxiv_id.strip()
        if norm_arxiv in arxiv_map:
            return arxiv_map[norm_arxiv]

    if title:
        norm_title = _normalize_title(title)
        if norm_title and norm_title in title_map:
            return title_map[norm_title]

    return None


def _compact_paper(
    raw: Dict[str, Any],
    doi_map: Dict[str, str],
    arxiv_map: Dict[str, str],
    title_map: Dict[str, str],
) -> Dict[str, Any]:
    """Project a raw Semantic Scholar paper row to the compact shape (spec
    §6): title, year, venue, doi, arxiv_id, citation_count, abstract<=400
    chars, plus the in_corpus cross-reference. Today's median response was
    3.8 K chars and p90 26 K with 45 client timeouts recorded -- this is the
    fix."""
    external_ids = raw.get("externalIds") or {}
    doi = external_ids.get("DOI")
    arxiv_id = external_ids.get("ArXiv")
    title = raw.get("title")

    return {
        "title": title,
        "year": raw.get("year"),
        "venue": raw.get("venue"),
        "doi": doi,
        "arxiv_id": arxiv_id,
        "citation_count": raw.get("citationCount"),
        "abstract": _truncate_abstract(raw.get("abstract")),
        "in_corpus": _match_in_corpus(doi, arxiv_id, title, doi_map, arxiv_map, title_map),
    }


def search(
    query: str,
    *,
    venue: str = "",
    year: str = "",
    fields_of_study: str = "",
    limit: int = 10,
) -> Dict[str, Any]:
    """Search Semantic Scholar for papers.

    Args:
        query: Free-text search query.
        venue: Optional exact venue name filter (comma-separated for
            multiple venues).
        year: Optional year filter, any form the Semantic Scholar Graph API
            accepts (a single year "2023", or a range "2019-2023",
            "2019-", "-2023").
        fields_of_study: Optional comma-separated field-of-study filter
            (e.g. "Computer Science,Engineering").
        limit: Number of results (1-100, default 10).

    Returns a structured {ok, data|error} dict — never a raw error body
    presented as success (F4). On success, `data["papers"]` rows are the
    compact projection with `in_corpus` attached (see `_compact_paper`).
    """
    limit = min(max(limit, 1), 100)
    params: Dict[str, Any] = {
        "query": query,
        "limit": limit,
        "fields": _S2_FIELDS,
    }
    if venue:
        params["venue"] = venue
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    result = _s2_get("paper/search", params)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}

    data = result["data"]
    raw_papers: List[Dict[str, Any]] = data.get("data", []) or []

    doi_map, arxiv_map, title_map = _corpus_lookup()
    compact_papers = [_compact_paper(p, doi_map, arxiv_map, title_map) for p in raw_papers]

    return {"ok": True, "data": {"total": data.get("total"), "papers": compact_papers}}


def _as_s2_paper_path(identifier: str) -> str:
    """Turn a DOI, bare/prefixed arXiv id, or raw Semantic Scholar paper id
    into the path segment the `/paper/{id}` endpoint expects (DOI:<doi>,
    ARXIV:<id>, or the identifier unchanged for anything else -- including
    an already-prefixed CorpusId:/MAG:/etc. or a bare S2 paper id)."""
    ident = (identifier or "").strip()

    lowered = ident.lower()
    if lowered.startswith("doi:"):
        return f"DOI:{ident.split(':', 1)[1].strip()}"
    if lowered.startswith("arxiv:"):
        return f"ARXIV:{ident.split(':', 1)[1].strip()}"
    # DOIs are "10.<registrant>/<suffix>" (DOI Handbook §2) -- a simple,
    # self-contained shape check (kept local rather than reusing
    # core.papers' private `_looks_like_doi`, per "keep it simple").
    if lowered.startswith("10.") and "/" in ident:
        return f"DOI:{ident}"
    if _ARXIV_IN_DOI_RE.fullmatch(ident):
        return f"ARXIV:{ident}"

    return ident


def get_paper(identifier: str) -> Dict[str, Any]:
    """Fetch a single Semantic Scholar paper by DOI, arXiv id, or S2 paper
    id (spec §6 `semantic_scholar_paper`, paired with `doi2content` misses:
    an agent that can't find a paper locally can still get its metadata and
    abstract). Same rate limiter, retry and cache path as `search()` --
    routed through `_s2_get`, no second HTTP path.

    Returns a structured {ok, data|error} dict; `data` is the same compact
    projection `search()` rows use, including `in_corpus`.
    """
    path = f"paper/{_as_s2_paper_path(identifier)}"
    params = {"fields": _S2_FIELDS}

    result = _s2_get(path, params)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error")}

    raw = result["data"]
    doi_map, arxiv_map, title_map = _corpus_lookup()
    return {"ok": True, "data": _compact_paper(raw, doi_map, arxiv_map, title_map)}
