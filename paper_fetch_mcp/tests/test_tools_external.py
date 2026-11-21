"""Tests for the `@mcp.tool` wrappers in tools/external.py: alphaxiv_paper_lookup,
alphaxiv_batch_lookup, semantic_scholar_search.

These wrappers must stay THIN (spec §5: "parse args, call core/*, respond()")
-- every test here mocks the `core.alphaxiv`/`core.s2` call the wrapper
delegates to, so nothing here ever makes a real HTTP call.

This is the new home for two properties test_server_v2_hotfix.py used to
check against the monolith's private `_alphaxiv_lookup` helper and its
`semantic_scholar_search` tool body directly:

  - alphaxiv_paper_lookup/alphaxiv_batch_lookup share the same lookup logic
    (now `core.alphaxiv.lookup`), each passing through its own max_chars.
  - semantic_scholar_search surfaces an upstream failure as a structured
    `kind: upstream_error` envelope, never a raw error string or (worse) the
    provider's raw failure body presented as if it were success.
"""

from __future__ import annotations

import sys
import pathlib
from unittest.mock import patch

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from paper_fetch_mcp.tools import external


def test_alphaxiv_paper_lookup_and_batch_lookup_share_core_alphaxiv_lookup():
    calls = []

    def fake_lookup(raw, max_chars=2000):
        calls.append((raw, max_chars))
        return {"status": "ok", "paper_id": raw, "overview": "x"}

    with patch.object(external.alphaxiv, "lookup", side_effect=fake_lookup):
        external.alphaxiv_paper_lookup("solo-id", max_chars=999)
        assert calls[-1] == ("solo-id", 999)

        calls.clear()
        external.alphaxiv_batch_lookup(["a", "b", "c"], max_chars=123)
        assert calls == [("a", 123), ("b", 123), ("c", 123)]


def test_alphaxiv_batch_lookup_caps_at_twenty_ids():
    calls = []

    def fake_lookup(raw, max_chars=2000):
        calls.append(raw)
        return {"status": "ok", "paper_id": raw}

    with patch.object(external.alphaxiv, "lookup", side_effect=fake_lookup):
        external.alphaxiv_batch_lookup([f"id{i}" for i in range(30)])

    assert len(calls) == 20


def test_semantic_scholar_search_returns_structured_error_on_upstream_failure():
    with patch.object(
        external.s2, "search", return_value={"ok": False, "error": "HTTP 429: nope"}
    ):
        out = external.semantic_scholar_search(query="x")

    data = yaml.safe_load(out)
    assert data["kind"] == "upstream_error"
    assert "429" in data["error"]


def test_semantic_scholar_search_returns_papers_list_on_success():
    with patch.object(
        external.s2,
        "search",
        return_value={"ok": True, "data": {"total": 1, "papers": [{"title": "A Paper"}]}},
    ):
        out = external.semantic_scholar_search(query="x")

    data = yaml.safe_load(out)
    assert data["total"] == 1
    assert data["papers"][0]["title"] == "A Paper"
    assert "meta" in data


def test_semantic_scholar_search_passes_year_and_fields_of_study_through():
    with patch.object(
        external.s2,
        "search",
        return_value={"ok": True, "data": {"total": 0, "papers": []}},
    ) as mock_search:
        external.semantic_scholar_search(
            query="x", year="2020-2023", fields_of_study="Computer Science"
        )

    mock_search.assert_called_once_with(
        "x", venue="", year="2020-2023", fields_of_study="Computer Science", limit=10
    )


def test_semantic_scholar_paper_returns_compact_data_on_success():
    with patch.object(
        external.s2,
        "get_paper",
        return_value={
            "ok": True,
            "data": {
                "title": "A Paper",
                "year": 2024,
                "venue": "TWC",
                "doi": "10.1109/TWC.2024.001",
                "arxiv_id": None,
                "citation_count": 3,
                "abstract": "abstract",
                "in_corpus": "smith2024deep",
            },
        },
    ) as mock_get_paper:
        out = external.semantic_scholar_paper(identifier="10.1109/TWC.2024.001")

    mock_get_paper.assert_called_once_with("10.1109/TWC.2024.001")
    data = yaml.safe_load(out)
    assert data["title"] == "A Paper"
    assert data["in_corpus"] == "smith2024deep"


def test_semantic_scholar_paper_returns_structured_error_on_upstream_failure():
    with patch.object(
        external.s2, "get_paper", return_value={"ok": False, "error": "HTTP 404: not found"}
    ):
        out = external.semantic_scholar_paper(identifier="nope")

    data = yaml.safe_load(out)
    assert data["kind"] == "upstream_error"
    assert "404" in data["error"]
