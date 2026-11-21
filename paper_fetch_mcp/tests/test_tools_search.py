"""Tests for paper_fetch_mcp/tools/search.py: the `search_metadata` tool
wrapper (argument clamping + pass-through to `core.papers.search_metadata`,
which has its own tests in test_core_papers.py).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from paper_fetch_mcp.tools import search as search_tools


@pytest.mark.parametrize("given, expected", [(0, 1), (20, 20), (500, 100)])
def test_search_metadata_clamps_limit(given, expected):
    with patch.object(
        search_tools.papers, "search_metadata", return_value={"results": [], "count": 0}
    ) as core:
        search_tools.search_metadata(query="semantic communication", limit=given)
    assert core.call_args.kwargs["limit"] == expected


def test_search_metadata_passes_filters_through_and_returns_yaml():
    rows = [{"citation_key": "Doe2024A", "title": "A Paper", "year": 2024}]
    with patch.object(
        search_tools.papers, "search_metadata", return_value={"results": rows, "count": 1}
    ) as core:
        out = search_tools.search_metadata(author="Doe", year_from=2020, year_to=2024)
    assert core.call_args.kwargs == {
        "query": None,
        "author": "Doe",
        "year_from": 2020,
        "year_to": 2024,
        "limit": 20,
    }
    data = yaml.safe_load(out)
    assert data["results"][0]["citation_key"] == "Doe2024A"
