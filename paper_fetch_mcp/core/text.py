"""Small text helpers shared across ``core/*`` modules.

No dependencies on any other ``paper_fetch_mcp`` module and no side effects
at import time.
"""

from __future__ import annotations

_DEFAULT_SNIPPET_CHARS = 1500


def snippet_truncate(text: str, limit: int = _DEFAULT_SNIPPET_CHARS) -> str:
    """Hard left-truncate `text` to `limit` chars, appending "…[truncated]"
    when it was actually cut. Passthrough (no marker) when `text` already
    fits."""
    if not text or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…[truncated]"
