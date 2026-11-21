"""AlphaXiv paper lookup: parse an ArXiv paper id out of arbitrary input and
fetch its AlphaXiv AI-generated overview (falling back to full text on 404).

Extracted verbatim (Phase 1, wave 1b) from the Phase-0-hotfixed
`paper_fetch_mcp/mcp_server_v2.py` (`_parse_paper_id`, `_alphaxiv_fetch`,
`_alphaxiv_lookup`, lines 814/832/843 as of the extraction).

No side effects at import time: no HTTP calls happen until `fetch()` /
`lookup()` are called.
"""

import re
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

from paper_fetch_mcp.core.text import snippet_truncate as _snippet

# `_snippet` above is `core.text.snippet_truncate`.


def _parse_paper_id(raw: str) -> Optional[str]:
    """Extract a normalised ArXiv paper ID from arbitrary user input."""
    raw = raw.strip()
    if re.match(r"^\d+\.\d+$", raw):
        return raw
    if re.match(r"^\d+\.\d+v\d+$", raw):
        return raw
    for pattern in [
        r"arxiv\.org/abs/([0-9]+\.[0-9]+(?:v\d+)?)",
        r"arxiv\.org/pdf/([0-9]+\.[0-9]+(?:v\d+)?)",
        r"alphaxiv\.org/(?:overview|abs)/([0-9]+\.[0-9]+(?:v\d+)?)",
    ]:
        m = re.search(pattern, raw)
        if m:
            return m.group(1)
    return None


def _alphaxiv_fetch(url: str) -> Tuple[str, int]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticRAGServer/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace"), resp.status
    except urllib.error.HTTPError as e:
        return "", e.code
    except Exception as e:
        return f"Network error: {e}", 0


def _alphaxiv_lookup(paper_id_or_url: str, max_chars: int = 15000) -> Dict[str, Any]:
    """Sync implementation shared by alphaxiv_paper_lookup and
    alphaxiv_batch_lookup — a plain dict, not YAML, so callers can compose it
    into a batch payload without a fragile yaml.safe_load(result) round trip.

    Truncates fetched overview/full_text to max_chars before returning.
    """
    paper_id = _parse_paper_id(paper_id_or_url)
    if not paper_id:
        return {
            "status": "error",
            "paper_id": paper_id_or_url,
            "error": (
                f"Could not parse an ArXiv ID from: '{paper_id_or_url}'. "
                "Expected a bare ID (e.g. 2401.12345), a versioned ID "
                "(e.g. 2401.12345v2), or an arxiv.org / alphaxiv.org URL."
            ),
        }

    overview_url = f"https://alphaxiv.org/overview/{paper_id}.md"
    fulltext_url = f"https://alphaxiv.org/abs/{paper_id}.md"

    overview, status = _alphaxiv_fetch(overview_url)
    if status == 200:
        return {
            "status": "ok",
            "paper_id": paper_id,
            "overview": _snippet(overview, max_chars),
            "source": "alphaxiv_overview",
        }

    fulltext, status = _alphaxiv_fetch(fulltext_url)
    if status == 200:
        return {
            "status": "ok",
            "paper_id": paper_id,
            "full_text": _snippet(fulltext, max_chars),
            "source": "alphaxiv_fulltext",
        }

    if status == 404:
        return {
            "status": "not_found",
            "paper_id": paper_id,
            "error": (
                "No AlphaXiv report available for this paper. "
                f"Direct PDF: https://arxiv.org/pdf/{paper_id}"
            ),
        }

    return {
        "status": "error",
        "paper_id": paper_id,
        "error": overview or f"HTTP {status}",
    }


def parse_paper_id(raw: str) -> Optional[str]:
    """Extract a normalised ArXiv paper ID from arbitrary user input."""
    return _parse_paper_id(raw)


def fetch(url: str) -> Tuple[str, int]:
    """Fetch a URL, returning (body_or_error_text, http_status). status is 0
    on a non-HTTPError exception (e.g. network failure)."""
    return _alphaxiv_fetch(url)


def lookup(paper_id_or_url: str, max_chars: int = 15000) -> Dict[str, Any]:
    """Look up an ArXiv paper via AlphaXiv.org: overview, falling back to
    full text on 404. Truncates the returned overview/full_text to
    max_chars."""
    return _alphaxiv_lookup(paper_id_or_url, max_chars=max_chars)
