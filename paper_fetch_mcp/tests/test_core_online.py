"""Tests for paper_fetch_mcp.core.online -- the network fallback behind a
`get_paper` corpus miss (publisher/arXiv fetch with the exit-node retry,
Semantic Scholar title resolution, and the online/miss envelopes).

`publisher.orchestrate_fetch` and `s2.search` are stubbed throughout; the
exit-node availability probe is stubbed too. No network, no DB.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from paper_fetch_mcp import config
from paper_fetch_mcp.core import online

LOCAL_MISS = {
    "found": False,
    "query": "whatever",
    "suggestions": [],
    "next": ["search_metadata", "semantic_scholar_search"],
}


def _orchestrate_yaml(**fields) -> str:
    base = {
        "status": "ok",
        "cache_hit": False,
        "source": "IEEE",
        "url": "https://ieeexplore.ieee.org/document/1",
        "fetched_at": "2026-09-14T00:00:00+00:00",
        "route": "direct",
        "html": "# Paper\n\n" + ("Body text. " * 50),
    }
    base.update(fields)
    return yaml.dump(base, sort_keys=False)


@pytest.fixture
def online_on(monkeypatch):
    """conftest disables the fallback for every test; opt back in and stub
    the exit-node probe so no socket is opened."""
    monkeypatch.setattr(config, "GET_PAPER_ONLINE_FALLBACK", True)
    with patch(
        "paper_fetch_mcp.core.exit_node.availability",
        return_value={
            "mode": "auto",
            "exit_node": "tung-lab-pc",
            "proxy_url": "socks5h://127.0.0.1:1056",
            "available": True,
            "reason": "stubbed",
        },
    ):
        yield


# ---------------------------------------------------------------------------
# classify_identifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier,expected",
    [
        ("10.1109/TWC.2024.001", {"kind": "doi", "doi": "10.1109/twc.2024.001"}),
        ("https://doi.org/10.1109/TWC.2024.001", {"kind": "doi", "doi": "10.1109/twc.2024.001"}),
        ("doi:10.1109/TWC.2024.001", {"kind": "doi", "doi": "10.1109/twc.2024.001"}),
        ("2301.12345", {"kind": "arxiv", "arxiv_id": "2301.12345"}),
        ("arXiv:2301.12345v2", {"kind": "arxiv", "arxiv_id": "2301.12345"}),
        ("https://arxiv.org/abs/2301.12345v1", {"kind": "arxiv", "arxiv_id": "2301.12345"}),
        ("https://arxiv.org/pdf/2301.12345", {"kind": "arxiv", "arxiv_id": "2301.12345"}),
        (
            "https://link.springer.com/article/10.1007/x",
            {"kind": "url", "url": "https://link.springer.com/article/10.1007/x"},
        ),
        ("Deep Joint Source-Channel Coding", {"kind": "title"}),
        ("AuthorX2023Great", {"kind": "title"}),
        ("", {"kind": "title"}),
    ],
)
def test_classify_identifier(identifier, expected):
    assert online.classify_identifier(identifier) == expected


def test_fetch_target_per_kind():
    assert online._fetch_target({"kind": "doi", "doi": "10.1/x"}) == "10.1/x"
    assert online._fetch_target({"kind": "arxiv", "arxiv_id": "2301.12345"}) == (
        "https://arxiv.org/abs/2301.12345"
    )
    assert online._fetch_target({"kind": "url", "url": "https://a/b"}) == "https://a/b"
    with pytest.raises(ValueError):
        online._fetch_target({"kind": "title"})


# ---------------------------------------------------------------------------
# get_paper_online: switched off
# ---------------------------------------------------------------------------


def test_disabled_returns_local_miss_untouched_but_annotated():
    # conftest has GET_PAPER_ONLINE_FALLBACK=False for this test.
    with patch("paper_fetch_mcp.core.online.publisher.orchestrate_fetch") as fetch, \
         patch("paper_fetch_mcp.core.online.s2.search") as search:
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    fetch.assert_not_called()
    search.assert_not_called()
    assert out["found"] is False
    assert out["suggestions"] == [] and out["next"] == LOCAL_MISS["next"]
    assert out["online"] == {"attempted": False, "reason": "GET_PAPER_ONLINE_FALLBACK=0"}
    assert "online" not in LOCAL_MISS  # input dict not mutated


# ---------------------------------------------------------------------------
# get_paper_online: DOI / arXiv / URL hits
# ---------------------------------------------------------------------------


def test_doi_hit_builds_online_envelope_with_route(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(route="exit_node"),
    ) as fetch:
        out = online.get_paper_online(
            "https://doi.org/10.1109/TWC.2024.001", local_miss=LOCAL_MISS, max_chars_per_paper=20
        )
    # normalised DOI is the fetch target (and thus the cached_html key)
    fetch.assert_called_once_with("10.1109/twc.2024.001")
    assert out["found"] is True
    assert out["source"] == "online"
    assert out["kind"] == "doi" and out["doi"] == "10.1109/twc.2024.001"
    assert out["arxiv_id"] is None
    assert out["url"] == "https://ieeexplore.ieee.org/document/1"
    assert out["publisher"] == "IEEE"
    assert out["status"] == "ok"
    assert out["route"] == "exit_node"
    assert out["content"] == "# Paper\n\nBody text. "  # 20-char window
    assert out["paging"]["returned_chars"] == 20
    assert out["paging"]["truncated"] is True
    assert "partial" not in out and "resolution" not in out and "next" not in out


def test_paging_offset_matches_local_contract(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(html="0123456789"),
    ):
        p1 = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS, max_chars_per_paper=4)
        p2 = online.get_paper_online(
            "10.1109/TWC.2024.001", local_miss=LOCAL_MISS, max_chars_per_paper=4, offset=4
        )
        p3 = online.get_paper_online(
            "10.1109/TWC.2024.001", local_miss=LOCAL_MISS, max_chars_per_paper=4, offset=8
        )
    assert (p1["content"], p2["content"], p3["content"]) == ("0123", "4567", "89")
    assert p1["paging"] == {"offset": 0, "returned_chars": 4, "total_chars": 10, "truncated": True}
    assert p3["paging"]["truncated"] is False


def test_arxiv_hit_targets_abs_page_and_suggests_ingest(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(
            source="Other", url="https://arxiv.org/abs/2301.12345", route="direct"
        ),
    ) as fetch:
        out = online.get_paper_online("arXiv:2301.12345v2", local_miss=LOCAL_MISS)
    fetch.assert_called_once_with("https://arxiv.org/abs/2301.12345")
    assert out["found"] is True and out["kind"] == "arxiv"
    assert out["arxiv_id"] == "2301.12345" and out["doi"] is None
    assert out["route"] == "direct"
    assert out["next"] == ["ingest_arxiv 2301.12345"]


def test_url_hit_is_fetched_as_is(online_on):
    url = "https://link.springer.com/article/10.1007/s11276-024-1"
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(source="Springer", url=url),
    ) as fetch:
        out = online.get_paper_online(url, local_miss=LOCAL_MISS)
    fetch.assert_called_once_with(url)
    assert out["found"] is True and out["kind"] == "url" and out["publisher"] == "Springer"


def test_cache_hit_reports_route_cache(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(status="cache_hit", cache_hit=True, route="cache"),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert out["status"] == "cache_hit" and out["route"] == "cache"


def test_legacy_cache_row_without_route_field_defaults_to_cache(online_on):
    """Rows cached before the `route` field existed come back without it."""
    raw = yaml.safe_load(_orchestrate_yaml(status="cache_hit", cache_hit=True))
    raw.pop("route")
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=yaml.dump(raw, sort_keys=False),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert out["route"] == "cache"


def test_s2_abstract_fallback_is_marked_partial(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(
            status="fallback",
            source="fallback",
            html="<p>Only the abstract.</p>",
            fallback_reason="direct: HTTP 403; via exit node tung-lab-pc: HTTP 403",
            route="exit_node",
        ),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert out["found"] is True and out["status"] == "fallback"
    assert out["partial"] is True
    assert "HTTP 403" in out["fallback_reason"]
    assert out["next"] == ["fetch_publisher_html 10.1109/twc.2024.001 force_refresh=true"]


# ---------------------------------------------------------------------------
# get_paper_online: failures keep the local miss envelope, annotated
# ---------------------------------------------------------------------------


def test_fetch_error_returns_annotated_miss(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(
            status="error",
            html="",
            error="requests fetch failed: direct: HTTP 403; via exit node tung-lab-pc: HTTP 403 | Abstract fallback also failed.",
        ),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert out["found"] is False
    assert out["suggestions"] == [] and out["next"] == LOCAL_MISS["next"]
    on = out["online"]
    assert on["attempted"] is True
    assert on["kind"] == "doi" and on["target"] == "10.1109/twc.2024.001"
    assert on["status"] == "error" and "HTTP 403" in on["error"]
    assert on["exit_node"] == {"name": "tung-lab-pc", "available": True, "reason": "stubbed"}


def test_orchestrate_raising_is_contained(online_on):
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        side_effect=RuntimeError("boom"),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert out["found"] is False
    assert out["online"]["status"] == "error" and "RuntimeError: boom" in out["online"]["error"]


def test_miss_reports_exit_node_unavailable_reason(monkeypatch):
    monkeypatch.setattr(config, "GET_PAPER_ONLINE_FALLBACK", True)
    with patch(
        "paper_fetch_mcp.core.exit_node.availability",
        return_value={
            "mode": "auto",
            "exit_node": "tung-lab-pc",
            "proxy_url": "x",
            "available": False,
            "reason": "exit-node proxy 127.0.0.1:1056 is not listening",
        },
    ), patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(status="error", html="", error="HTTP 403"),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert out["online"]["exit_node"]["available"] is False
    assert "not listening" in out["online"]["exit_node"]["reason"]


# ---------------------------------------------------------------------------
# get_paper_online: title -> Semantic Scholar -> DOI / arXiv / in_corpus
# ---------------------------------------------------------------------------


def _s2_ok(*rows):
    return {"ok": True, "data": {"total": len(rows), "papers": list(rows)}}


def test_title_resolves_via_s2_exact_match_then_fetches_doi(online_on):
    rows = [
        {"title": "Other Paper", "year": 2020, "doi": "10.9/other", "arxiv_id": None, "in_corpus": None},
        {"title": "Deep JSCC for Wireless", "year": 2019, "doi": "10.1109/TCCN.2019.1", "arxiv_id": "1809.01733", "in_corpus": None},
    ]
    with patch("paper_fetch_mcp.core.online.s2.search", return_value=_s2_ok(*rows)) as search, \
         patch(
             "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
             return_value=_orchestrate_yaml(),
         ) as fetch:
        out = online.get_paper_online("deep jscc for wireless", local_miss=LOCAL_MISS)
    search.assert_called_once_with("deep jscc for wireless", limit=3)
    fetch.assert_called_once_with("10.1109/tccn.2019.1")  # DOI preferred over arXiv id
    assert out["found"] is True and out["kind"] == "doi"
    assert out["resolution"] == {
        "via": "semantic_scholar",
        "title": "Deep JSCC for Wireless",
        "year": 2019,
        "match": "exact_title",
    }


def test_title_top_hit_without_doi_falls_back_to_arxiv(online_on):
    rows = [{"title": "Some Preprint", "year": 2024, "doi": None, "arxiv_id": "2401.00001", "in_corpus": None}]
    with patch("paper_fetch_mcp.core.online.s2.search", return_value=_s2_ok(*rows)), \
         patch(
             "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
             return_value=_orchestrate_yaml(source="Other", url="https://arxiv.org/abs/2401.00001"),
         ) as fetch:
        out = online.get_paper_online("some preprint about things", local_miss=LOCAL_MISS)
    fetch.assert_called_once_with("https://arxiv.org/abs/2401.00001")
    assert out["kind"] == "arxiv" and out["resolution"]["match"] == "top_hit"
    assert out["next"] == ["ingest_arxiv 2401.00001"]


def test_title_in_corpus_serves_local_paper(online_on):
    rows = [{"title": "A Great Paper", "year": 2023, "doi": "10.4321/x", "arxiv_id": None, "in_corpus": "AuthorX2023Great"}]
    local_hit = {
        "found": True,
        "citation_key": "AuthorX2023Great",
        "id": "2301.12345",
        "doi": "10.4321/x",
        "title": "A Great Paper",
        "content": "Intro body.",
        "paging": {"offset": 0, "returned_chars": 11, "total_chars": 11, "truncated": False},
    }
    with patch("paper_fetch_mcp.core.online.s2.search", return_value=_s2_ok(*rows)), \
         patch("paper_fetch_mcp.core.online.papers.get_paper", return_value=dict(local_hit)) as local, \
         patch("paper_fetch_mcp.core.online.publisher.orchestrate_fetch") as fetch:
        out = online.get_paper_online(
            "a great paper", local_miss=LOCAL_MISS, max_chars_per_paper=500, offset=0
        )
    local.assert_called_once_with(
        "AuthorX2023Great", max_chars_per_paper=500, offset=0, save_path=""
    )
    fetch.assert_not_called()
    assert out["found"] is True and out["citation_key"] == "AuthorX2023Great"
    assert "source" not in out  # a corpus hit, not an online one
    assert out["query"] == "a great paper"
    assert out["resolution"]["via"] == "semantic_scholar"


def test_title_no_s2_match_returns_unresolved_miss(online_on):
    with patch("paper_fetch_mcp.core.online.s2.search", return_value=_s2_ok()), \
         patch("paper_fetch_mcp.core.online.publisher.orchestrate_fetch") as fetch:
        out = online.get_paper_online("nonexistent-xyz-987", local_miss=LOCAL_MISS)
    fetch.assert_not_called()
    assert out["found"] is False
    assert out["online"]["kind"] == "title" and out["online"]["status"] == "unresolved"
    assert "no Semantic Scholar match" in out["online"]["error"]


def test_title_s2_error_returns_unresolved_miss(online_on):
    with patch(
        "paper_fetch_mcp.core.online.s2.search",
        return_value={"ok": False, "error": "HTTP 429 after 4 attempts"},
    ):
        out = online.get_paper_online("some title", local_miss=LOCAL_MISS)
    assert out["found"] is False and out["online"]["status"] == "unresolved"


def test_title_hit_without_doi_or_arxiv_is_unresolved(online_on):
    rows = [{"title": "Book Chapter", "year": 2010, "doi": None, "arxiv_id": None, "in_corpus": None}]
    with patch("paper_fetch_mcp.core.online.s2.search", return_value=_s2_ok(*rows)), \
         patch("paper_fetch_mcp.core.online.publisher.orchestrate_fetch") as fetch:
        out = online.get_paper_online("book chapter", local_miss=LOCAL_MISS)
    fetch.assert_not_called()
    assert out["found"] is False
    assert "neither a DOI nor an arXiv id" in out["online"]["error"]
    assert out["online"]["resolution"]["title"] == "Book Chapter"


# ---------------------------------------------------------------------------
# save_path: write the fetched paper to disk
# ---------------------------------------------------------------------------


def test_save_path_writes_full_fetched_content_not_the_window(online_on, tmp_path):
    target = tmp_path / "fetched.md"
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(html="0123456789"),
    ):
        out = online.get_paper_online(
            "10.1109/TWC.2024.001",
            local_miss=LOCAL_MISS,
            max_chars_per_paper=4,
            save_path=str(target),
        )
    assert out["content"] == "0123"  # result still windowed
    assert target.read_text(encoding="utf-8") == "0123456789"  # file is whole
    assert out["saved"] == {
        "path": str(target),
        "chars": 10,
        "bytes": 10,
        "overwrote": False,
    }


def test_save_path_directory_names_file_after_arxiv_id(online_on, tmp_path):
    with (
        patch(
            "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
            return_value=_orchestrate_yaml(url="https://arxiv.org/abs/2301.12345"),
        ),
        patch("paper_fetch_mcp.core.papers.PAPER_SAVE_DIR", tmp_path / "saved"),
    ):
        out = online.get_paper_online(
            "arXiv:2301.12345v2", local_miss=LOCAL_MISS, save_path="online/"
        )
    expected = tmp_path / "saved" / "online" / "2301.12345.md"
    assert out["saved"]["path"] == str(expected)
    assert expected.exists()


def test_save_path_is_passed_through_to_a_local_corpus_hit(online_on, tmp_path):
    """A title that Semantic Scholar resolves to a paper already in the
    corpus is served locally -- save_path must still reach that local
    `papers.get_paper` call rather than being dropped."""
    rows = [
        {
            "title": "A Great Paper",
            "year": 2023,
            "doi": "10.4321/x",
            "arxiv_id": None,
            "in_corpus": "AuthorX2023Great",
        }
    ]
    local_hit = {
        "found": True,
        "citation_key": "AuthorX2023Great",
        "content": "Intro body.",
        "paging": {"offset": 0, "returned_chars": 11, "total_chars": 11, "truncated": False},
        "saved": {"path": "/tmp/x.md", "chars": 11, "bytes": 11, "overwrote": False},
    }
    with (
        patch("paper_fetch_mcp.core.online.s2.search", return_value=_s2_ok(*rows)),
        patch(
            "paper_fetch_mcp.core.online.papers.get_paper", return_value=dict(local_hit)
        ) as local,
    ):
        out = online.get_paper_online(
            "a great paper", local_miss=LOCAL_MISS, save_path="/tmp/x.md"
        )
    local.assert_called_once_with(
        "AuthorX2023Great", max_chars_per_paper=12000, offset=0, save_path="/tmp/x.md"
    )
    assert out["saved"]["path"] == "/tmp/x.md"


def test_no_save_path_means_no_saved_block(online_on, tmp_path):
    with (
        patch(
            "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
            return_value=_orchestrate_yaml(),
        ),
        patch("paper_fetch_mcp.core.papers.PAPER_SAVE_DIR", tmp_path / "saved"),
    ):
        out = online.get_paper_online("10.1109/TWC.2024.001", local_miss=LOCAL_MISS)
    assert "saved" not in out
    assert not (tmp_path / "saved").exists()


def test_online_miss_does_not_save(online_on, tmp_path):
    target = tmp_path / "nothing.md"
    with patch(
        "paper_fetch_mcp.core.online.publisher.orchestrate_fetch",
        return_value=_orchestrate_yaml(status="error", html="", error="403 paywall"),
    ):
        out = online.get_paper_online(
            "10.1109/TWC.2024.001", local_miss=LOCAL_MISS, save_path=str(target)
        )
    assert out["found"] is False
    assert "saved" not in out
    assert not target.exists()
