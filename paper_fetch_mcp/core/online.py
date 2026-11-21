"""Online fallback for `get_paper` misses.

`core.papers.get_paper` is purely local: it resolves a citation key, DOI,
arXiv id or title against papers.db + mistral_output and returns a
`{found: false, ...}` envelope when the paper is not in the corpus. This
module is what `tools/content.get_paper` calls on that miss: it fetches
the paper from the network instead, with the exit-node retry in
`core.exit_node` behind every request, so a paper this host's own network
cannot reach (paywalled publisher page, blocked site) is still returned
when the Tailscale exit node (`ubuntu-lab` by default) can reach it.

Resolution (`classify_identifier`):

* DOI in any form           -> publisher page via `core.publisher`
                               (doi.org redirect -> requests-first fetch ->
                               exit-node retry -> markdown; cached in
                               `cached_html` keyed on the normalised DOI)
* arXiv id in any form      -> https://arxiv.org/abs/<id>, which
                               `core.publisher` turns into the full text:
                               arXiv's HTML version, else the PDF's text
                               (`next` points at `ingest_arxiv <id>` to add
                               it to the corpus)
* http(s) URL               -> fetched as-is
* anything else (a title)   -> Semantic Scholar search first. If the top
                               hit is already `in_corpus` under a different
                               key the LOCAL paper is returned; otherwise
                               its DOI / arXiv id feeds the two cases above.

Hit envelope (same `content`/`paging` contract as the local one, plus
provenance so an agent can tell a network fetch from a corpus hit):

    {found: true, source: "online", query, kind, doi, arxiv_id, url,
     publisher, status: ok|cache_hit|fallback, route: direct|exit_node|cache,
     [fallback_reason, partial: true], [resolution: {via, title, year,
     match}], content, paging: {...}, [saved: {path, chars, bytes,
     overwrote}], next: [...]}

Miss envelope: the local `{found: false, query, suggestions, next}` with an
added `online: {attempted, kind, target, status, error, exit_node}` block
saying what was tried and why it did not work -- including whether the
exit node was available at the time.

The whole fallback is switched off with `GET_PAPER_ONLINE_FALLBACK=0`
(`config`), in which case the local miss envelope is returned untouched
apart from `online: {attempted: false, reason: ...}`.

No side effects at import time; no network until `get_paper_online` runs.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import yaml

from paper_fetch_mcp import config
from paper_fetch_mcp.core import exit_node
from paper_fetch_mcp.core import papers
from paper_fetch_mcp.core import publisher
from paper_fetch_mcp.core import s2

logger = logging.getLogger(__name__)

_BARE_ARXIV_RE = re.compile(r"^(?:arxiv:\s*)?(\d{4}\.\d{4,5})(?:v\d+)?$", re.IGNORECASE)
_S2_TITLE_LIMIT = 3


# ---------------------------------------------------------------------------
# Identifier classification
# ---------------------------------------------------------------------------


def classify_identifier(identifier: str) -> Dict[str, Any]:
    """Decide how to fetch ``identifier`` from the network.

    Returns ``{kind: doi|arxiv|url|title, doi?, arxiv_id?, url?}``.
    """
    ident = (identifier or "").strip()
    if not ident:
        return {"kind": "title"}

    lower = ident.lower()
    if lower.startswith(("http://", "https://")):
        if "arxiv.org/" in lower:
            arxiv_id = papers.extract_arxiv_id(ident)
            if arxiv_id:
                return {"kind": "arxiv", "arxiv_id": arxiv_id}
        if "doi.org/" in lower:
            doi = papers.normalize_doi(ident)
            if papers.looks_like_doi(doi):
                return {"kind": "doi", "doi": doi}
        return {"kind": "url", "url": ident}

    if papers.looks_like_doi(ident):
        return {"kind": "doi", "doi": papers.normalize_doi(ident)}

    m = _BARE_ARXIV_RE.match(ident)
    if m:
        return {"kind": "arxiv", "arxiv_id": m.group(1)}

    return {"kind": "title"}


def _fetch_target(info: Dict[str, Any]) -> str:
    """The identifier handed to `publisher.orchestrate_fetch` (which is also
    its `cached_html` cache key)."""
    kind = info["kind"]
    if kind == "doi":
        return info["doi"]
    if kind == "arxiv":
        return f"https://arxiv.org/abs/{info['arxiv_id']}"
    if kind == "url":
        return info["url"]
    raise ValueError(f"no fetch target for kind={kind!r}")


# ---------------------------------------------------------------------------
# Title -> DOI / arXiv id via Semantic Scholar
# ---------------------------------------------------------------------------


def resolve_title_online(title: str) -> Optional[Dict[str, Any]]:
    """Look ``title`` up on Semantic Scholar and return the best compact row
    (see `core.s2._compact_paper`: title, year, doi, arxiv_id, in_corpus,
    ...) tagged with ``match: exact_title|top_hit``, or ``None``.

    An exact (case-insensitive) title match among the first few hits wins;
    otherwise the top hit is taken and labelled as such so the caller
    surfaces the resolved title for the agent to verify.
    """
    result = s2.search(title, limit=_S2_TITLE_LIMIT)
    if not result.get("ok"):
        logger.info("S2 title resolution failed for %r: %s", title, result.get("error"))
        return None
    rows: List[Dict[str, Any]] = (result.get("data") or {}).get("papers") or []
    if not rows:
        return None

    wanted = title.strip().lower()
    for row in rows:
        if (row.get("title") or "").strip().lower() == wanted:
            return {**row, "match": "exact_title"}
    return {**rows[0], "match": "top_hit"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _miss(local_miss: Dict[str, Any], **online: Any) -> Dict[str, Any]:
    out = dict(local_miss)
    out["online"] = {"attempted": True, **online}
    return out


def _exit_node_summary() -> Dict[str, Any]:
    avail = exit_node.availability()
    return {
        "name": avail.get("exit_node"),
        "available": avail.get("available", False),
        "reason": avail.get("reason"),
    }


def get_paper_online(
    identifier: str,
    *,
    local_miss: Dict[str, Any],
    max_chars_per_paper: int = 12000,
    offset: int = 0,
    save_path: str = "",
) -> Dict[str, Any]:
    """Fetch ``identifier`` from the network after a local-corpus miss.

    ``local_miss`` is the `{found: false, ...}` envelope `core.papers.get_paper`
    produced; it is returned (annotated) when the online path fails too.
    See the module docstring for the envelope shapes.

    ``save_path`` is passed through to ``core.papers.save_paper_content``:
    the FULL fetched content (not the returned window) is written there and
    the hit envelope carries the ``saved: {...}`` report. A failed write is
    reported in that block and does not fail the fetch.
    """
    identifier = (identifier or "").strip()
    if not config.GET_PAPER_ONLINE_FALLBACK:
        out = dict(local_miss)
        out["online"] = {"attempted": False, "reason": "GET_PAPER_ONLINE_FALLBACK=0"}
        return out

    info = classify_identifier(identifier)
    resolution: Optional[Dict[str, Any]] = None

    if info["kind"] == "title":
        hit = resolve_title_online(identifier)
        if hit is None:
            return _miss(
                local_miss,
                kind="title",
                status="unresolved",
                error="no Semantic Scholar match for the title",
                exit_node=_exit_node_summary(),
            )
        resolution = {
            "via": "semantic_scholar",
            "title": hit.get("title"),
            "year": hit.get("year"),
            "match": hit.get("match"),
        }
        in_corpus = hit.get("in_corpus")
        if in_corpus:
            # The corpus has it under a key the title search did not reach --
            # serve the local copy rather than re-fetching it.
            local = papers.get_paper(
                in_corpus,
                max_chars_per_paper=max_chars_per_paper,
                offset=offset,
                save_path=save_path,
            )
            if local.get("found"):
                local["query"] = identifier
                local["resolution"] = resolution
                return local
        if hit.get("doi"):
            info = {"kind": "doi", "doi": papers.normalize_doi(hit["doi"])}
        elif hit.get("arxiv_id"):
            info = {"kind": "arxiv", "arxiv_id": hit["arxiv_id"]}
        else:
            return _miss(
                local_miss,
                kind="title",
                status="unresolved",
                error="Semantic Scholar match carries neither a DOI nor an arXiv id",
                resolution=resolution,
                exit_node=_exit_node_summary(),
            )

    target = _fetch_target(info)
    logger.info("get_paper online fallback for %r -> %s %s", identifier, info["kind"], target)
    try:
        raw = publisher.orchestrate_fetch(target)
        res: Dict[str, Any] = yaml.safe_load(raw) or {}
    except Exception as e:  # orchestrate_fetch is designed never to raise; belt and braces
        logger.warning("online fetch of %s raised: %s", target, e)
        res = {"status": "error", "error": f"{type(e).__name__}: {e}"}

    status = res.get("status")
    content = res.get("html") or ""
    if status in ("ok", "cache_hit", "fallback") and content:
        window, paging = papers.page_window(content, offset, max_chars_per_paper)
        route = res.get("route") or ("cache" if res.get("cache_hit") else "direct")
        envelope: Dict[str, Any] = {
            "found": True,
            "source": "online",
            "query": identifier,
            "kind": info["kind"],
            "doi": info.get("doi"),
            "arxiv_id": info.get("arxiv_id"),
            "url": res.get("url"),
            "publisher": res.get("source"),
            "status": status,
            "route": route,
            "content": window,
            "paging": paging,
        }
        if (save_path or "").strip():
            envelope["saved"] = papers.save_paper_content(
                content,
                save_path,
                default_stem=papers.save_stem(
                    info.get("arxiv_id"),
                    info.get("doi"),
                    (resolution or {}).get("title"),
                    identifier,
                ),
            )
        if status == "fallback":
            envelope["partial"] = True
            envelope["fallback_reason"] = res.get("fallback_reason")
        if resolution:
            envelope["resolution"] = resolution
        next_actions: List[str] = []
        if info.get("arxiv_id"):
            next_actions.append(f"ingest_arxiv {info['arxiv_id']}")
        if status == "fallback" and info.get("doi"):
            next_actions.append(f"fetch_publisher_html {info['doi']} force_refresh=true")
        if next_actions:
            envelope["next"] = next_actions
        return envelope

    return _miss(
        local_miss,
        kind=info["kind"],
        target=target,
        status="error",
        error=res.get("error") or f"online fetch returned status={status!r} with no content",
        resolution=resolution,
        exit_node=_exit_node_summary(),
    )
