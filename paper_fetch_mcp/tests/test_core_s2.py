"""Tests for paper_fetch_mcp.core.s2: the Semantic Scholar client.

These lock in the Phase 0 fix for F4 (0.5 spec): a process-wide 1 req/s rate
limiter, <=3 retries honouring Retry-After, and a 1h/256-entry TTL cache. A
regression in any of these brings back the 29 silent HTTP 429s mined from
production transcripts. No real network calls -- requests_mock intercepts
everything.

Phase 2 additions (spec §6/§8): compact projection (title, year, venue,
doi, arxiv_id, citation_count, abstract<=400 chars) and the `in_corpus`
cross-reference against the local corpus, plus `get_paper()` (backing the
new `semantic_scholar_paper` tool). The corpus cross-reference is tested
against a small temp SQLite DB (patched in via `core.papers.DB_PATH`, same
convention as tests/test_core_papers.py) -- never the live
pre_processing_v2/papers.db.
"""

import sqlite3
import threading
import time
from unittest.mock import patch

import pytest
import requests_mock

from paper_fetch_mcp.core import s2


@pytest.fixture(autouse=True)
def _reset_s2_module_state():
    """Each test gets a clean rate-limiter clock and empty cache -- module
    globals persist across tests otherwise."""
    s2._s2_last_call_monotonic = 0.0
    s2._s2_cache.clear()
    yield
    s2._s2_last_call_monotonic = 0.0
    s2._s2_cache.clear()


@pytest.fixture
def tmp_corpus_db(tmp_path):
    """A small temp corpus DB for the in_corpus cross-reference tests.
    Mirrors tests/test_core_papers.py's `tmp_db` fixture/schema, plus rows
    that exercise each match path (DOI, arXiv-via-DOI, title)."""
    db_path = tmp_path / "test_papers.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            publisher TEXT,
            booktitle_journal TEXT,
            year INTEGER,
            file TEXT,
            author TEXT,
            doi TEXT,
            citation_key TEXT,
            abstract TEXT,
            sections JSON
        )
        """
    )
    conn.executemany(
        "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "doi-paper", "A DOI-Matched Paper", "IEEE", "TWC", 2024,
                "doi-paper", "Smith", "10.1109/TWC.2024.001",
                "smithDoiMatched2024", "abstract", "[]",
            ),
            (
                "arxiv-paper", "An ArXiv-Sourced Paper", None, None, 2025,
                "arxiv-paper", "Zhu", "10.48550/arXiv.2507.11913",
                "zhuArxivMatched2025", "abstract", "[]",
            ),
            (
                "title-paper", "A Wholly Unique Title About Widgets", None, None, 2022,
                "title-paper", "Lee", None,
                "leeTitleMatched2022", "abstract", "[]",
            ),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def _search_url():
    return f"{s2._S2_BASE_URL}/paper/search"


def _paper_url(path_segment):
    return f"{s2._S2_BASE_URL}/paper/{path_segment}"


# ---------------------------------------------------------------------------
# Rate limiter / retry / cache (Phase 0 fix -- must keep passing unchanged)
# ---------------------------------------------------------------------------


def test_search_success_returns_ok_dict():
    with requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={"total": 1, "data": [{"title": "A Paper"}]},
            status_code=200,
        )
        result = s2.search("attention is all you need")

    assert result["ok"] is True
    assert result["data"]["total"] == 1
    assert result["data"]["papers"][0]["title"] == "A Paper"


def test_429_with_retry_after_triggers_retry_not_returned_as_success():
    with requests_mock.Mocker() as m, patch("time.sleep"):
        m.get(
            _search_url(),
            [
                {"status_code": 429, "headers": {"Retry-After": "0"}, "text": "slow down"},
                {"status_code": 200, "json": {"total": 1, "data": [{"title": "ok"}]}},
            ],
        )
        result = s2.search("query")

    assert result["ok"] is True
    assert result["data"]["papers"][0]["title"] == "ok"
    assert m.call_count == 2  # one 429, one successful retry


def test_429_raw_body_never_presented_as_success_after_exhausting_retries():
    with requests_mock.Mocker() as m, patch("time.sleep"):
        m.get(_search_url(), status_code=429, headers={"Retry-After": "0"}, text="rate limited")
        result = s2.search("query")

    assert result["ok"] is False
    assert "429" in result["error"]
    assert m.call_count == s2._S2_MAX_ATTEMPTS


def test_non_retryable_4xx_fails_immediately_without_retrying():
    with requests_mock.Mocker() as m:
        m.get(_search_url(), status_code=400, text="bad request")
        result = s2.search("query")

    assert result["ok"] is False
    assert m.call_count == 1  # no retries for a non-retryable 4xx


def test_cache_hit_avoids_second_http_call():
    with requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={"total": 1, "data": [{"title": "cached"}]},
            status_code=200,
        )
        first = s2.search("repeat query", limit=5)
        second = s2.search("repeat query", limit=5)

    assert first == second
    assert m.call_count == 1  # second call served entirely from cache


def test_cache_ttl_expiry_forces_a_second_http_call():
    with requests_mock.Mocker() as m, patch("time.sleep"):
        m.get(
            _search_url(),
            [
                {"json": {"total": 1, "data": [{"title": "first"}]}, "status_code": 200},
                {"json": {"total": 1, "data": [{"title": "second"}]}, "status_code": 200},
            ],
        )
        first = s2.search("ttl query")

        # Force the cached entry to look older than the TTL without sleeping.
        key = s2._s2_cache_key(
            "paper/search",
            {
                "query": "ttl query",
                "limit": 10,
                "fields": s2._S2_FIELDS,
            },
        )
        ts, value = s2._s2_cache[key]
        s2._s2_cache[key] = (ts - s2._S2_CACHE_TTL_S - 1, value)

        second = s2.search("ttl query")

    assert first["data"]["papers"][0]["title"] == "first"
    assert second["data"]["papers"][0]["title"] == "second"
    assert m.call_count == 2


def test_rate_limiter_enforces_min_interval_across_threads():
    s2._s2_last_call_monotonic = 0.0
    call_times = []
    lock = threading.Lock()

    def worker():
        s2._s2_rate_limit_wait()
        with lock:
            call_times.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    call_times.sort()
    for earlier, later in zip(call_times, call_times[1:]):
        assert later - earlier >= s2._S2_MIN_INTERVAL_S - 0.05  # small scheduling slack


def test_search_clamps_limit_to_valid_range():
    with requests_mock.Mocker() as m:
        m.get(_search_url(), json={"total": 0, "data": []}, status_code=200)
        s2.search("q", limit=500)
        assert m.last_request.qs["limit"] == ["100"]

        s2.search("q2", limit=0)
        assert m.last_request.qs["limit"] == ["1"]


# ---------------------------------------------------------------------------
# search(): new year/fields_of_study params, compact projection
# ---------------------------------------------------------------------------


def test_search_passes_year_and_fields_of_study_params():
    with requests_mock.Mocker() as m:
        m.get(_search_url(), json={"total": 0, "data": []}, status_code=200)
        s2.search("q", year="2019-2023", fields_of_study="Computer Science,Engineering")

    assert m.last_request.qs["year"] == ["2019-2023"]
    assert m.last_request.qs["fieldsofstudy"] == ["computer science,engineering"]


def test_search_omits_year_and_fields_of_study_when_not_given():
    with requests_mock.Mocker() as m:
        m.get(_search_url(), json={"total": 0, "data": []}, status_code=200)
        s2.search("q")

    assert "year" not in m.last_request.qs
    assert "fieldsofstudy" not in m.last_request.qs


def test_search_projects_compact_fields():
    with requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={
                "total": 1,
                "data": [
                    {
                        "paperId": "abc123",
                        "title": "Some Paper",
                        "year": 2024,
                        "venue": "TWC",
                        "citationCount": 42,
                        "externalIds": {"DOI": "10.1109/TWC.2024.999", "ArXiv": "2401.00001"},
                        "abstract": "short abstract",
                    }
                ],
            },
            status_code=200,
        )
        result = s2.search("q")

    row = result["data"]["papers"][0]
    assert row == {
        "title": "Some Paper",
        "year": 2024,
        "venue": "TWC",
        "doi": "10.1109/TWC.2024.999",
        "arxiv_id": "2401.00001",
        "citation_count": 42,
        "abstract": "short abstract",
        "in_corpus": None,
    }
    # paperId is not part of the compact projection (spec §6).
    assert "paperId" not in row


def test_search_truncates_abstract_to_400_chars():
    long_abstract = "x" * 1000
    with requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={"total": 1, "data": [{"title": "t", "abstract": long_abstract}]},
            status_code=200,
        )
        result = s2.search("q")

    abstract = result["data"]["papers"][0]["abstract"]
    assert len(abstract) <= 400
    assert abstract.endswith("…")


def test_search_leaves_short_abstract_untouched():
    with requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={"total": 1, "data": [{"title": "t", "abstract": "short"}]},
            status_code=200,
        )
        result = s2.search("q")

    assert result["data"]["papers"][0]["abstract"] == "short"


# ---------------------------------------------------------------------------
# in_corpus cross-reference: DOI, arXiv (via DOI), title match paths + miss
# ---------------------------------------------------------------------------


def test_in_corpus_matches_by_doi(tmp_corpus_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db), requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={
                "total": 1,
                "data": [
                    {
                        "title": "A DOI-Matched Paper (different casing)",
                        "externalIds": {"DOI": "https://doi.org/10.1109/TWC.2024.001"},
                    }
                ],
            },
            status_code=200,
        )
        result = s2.search("q")

    assert result["data"]["papers"][0]["in_corpus"] == "smithDoiMatched2024"


def test_in_corpus_matches_by_arxiv_id(tmp_corpus_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db), requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={
                "total": 1,
                "data": [
                    {
                        "title": "A totally different title",
                        "externalIds": {"ArXiv": "2507.11913"},
                    }
                ],
            },
            status_code=200,
        )
        result = s2.search("q")

    assert result["data"]["papers"][0]["in_corpus"] == "zhuArxivMatched2025"


def test_in_corpus_matches_by_title_when_no_doi_or_arxiv(tmp_corpus_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db), requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={
                "total": 1,
                "data": [
                    # Punctuation/case differ from the corpus title; the
                    # normalised (alnum-only, lowercased) comparison should
                    # still match.
                    {"title": "A Wholly-Unique Title, About Widgets!"}
                ],
            },
            status_code=200,
        )
        result = s2.search("q")

    assert result["data"]["papers"][0]["in_corpus"] == "leeTitleMatched2022"


def test_in_corpus_is_null_for_a_paper_not_in_the_corpus(tmp_corpus_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db), requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={
                "total": 1,
                "data": [
                    {
                        "title": "Nothing Like Any Corpus Paper",
                        "externalIds": {"DOI": "10.9999/not.in.corpus"},
                    }
                ],
            },
            status_code=200,
        )
        result = s2.search("q")

    assert result["data"]["papers"][0]["in_corpus"] is None


def test_in_corpus_prefers_doi_over_title_when_both_could_match(tmp_corpus_db):
    """DOI first, then arXiv id, then title (task spec order)."""
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db), requests_mock.Mocker() as m:
        m.get(
            _search_url(),
            json={
                "total": 1,
                "data": [
                    {
                        # Title matches the arxiv-paper row, but DOI matches
                        # the doi-paper row -- DOI must win.
                        "title": "An ArXiv-Sourced Paper",
                        "externalIds": {"DOI": "10.1109/TWC.2024.001"},
                    }
                ],
            },
            status_code=200,
        )
        result = s2.search("q")

    assert result["data"]["papers"][0]["in_corpus"] == "smithDoiMatched2024"


def test_corpus_lookup_is_built_once_per_call_not_per_row(tmp_corpus_db):
    """Cheap matching: one DB round trip per search() call, regardless of
    how many result rows it returns (the corpus is 1,648 papers)."""
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db):
        with patch.object(s2, "_corpus_lookup", wraps=s2._corpus_lookup) as spy:
            with requests_mock.Mocker() as m:
                m.get(
                    _search_url(),
                    json={
                        "total": 3,
                        "data": [{"title": f"paper {i}"} for i in range(3)],
                    },
                    status_code=200,
                )
                s2.search("q")

    assert spy.call_count == 1


def test_in_corpus_degrades_to_null_on_db_error():
    """A DB error must not fail the whole search -- in_corpus just can't be
    determined."""
    with patch(
        "paper_fetch_mcp.core.papers.get_db_connection", side_effect=RuntimeError("boom")
    ):
        with requests_mock.Mocker() as m:
            m.get(
                _search_url(),
                json={"total": 1, "data": [{"title": "t"}]},
                status_code=200,
            )
            result = s2.search("q")

    assert result["ok"] is True
    assert result["data"]["papers"][0]["in_corpus"] is None


# ---------------------------------------------------------------------------
# get_paper(): identifier routing + compact projection + in_corpus
# ---------------------------------------------------------------------------


def test_get_paper_routes_doi_identifier():
    with requests_mock.Mocker() as m:
        m.get(_paper_url("DOI:10.1109/TWC.2024.001"), json={"title": "t"}, status_code=200)
        result = s2.get_paper("10.1109/TWC.2024.001")

    assert result["ok"] is True
    assert m.called


def test_get_paper_routes_bare_arxiv_id():
    with requests_mock.Mocker() as m:
        m.get(_paper_url("ARXIV:2301.12345"), json={"title": "t"}, status_code=200)
        result = s2.get_paper("2301.12345")

    assert result["ok"] is True
    assert m.called


def test_get_paper_routes_prefixed_arxiv_identifier():
    with requests_mock.Mocker() as m:
        m.get(_paper_url("ARXIV:2301.12345"), json={"title": "t"}, status_code=200)
        result = s2.get_paper("arXiv:2301.12345")

    assert result["ok"] is True
    assert m.called


def test_get_paper_passes_through_raw_s2_id_unchanged():
    with requests_mock.Mocker() as m:
        m.get(_paper_url("649def34f8be52c8b66281af98ae884c09aef38b9"), json={"title": "t"}, status_code=200)
        result = s2.get_paper("649def34f8be52c8b66281af98ae884c09aef38b9")

    assert result["ok"] is True
    assert m.called


def test_get_paper_returns_compact_projection_with_in_corpus(tmp_corpus_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_corpus_db), requests_mock.Mocker() as m:
        m.get(
            _paper_url("DOI:10.1109/TWC.2024.001"),
            json={
                "title": "A DOI-Matched Paper",
                "year": 2024,
                "venue": "TWC",
                "citationCount": 5,
                "externalIds": {"DOI": "10.1109/TWC.2024.001"},
                "abstract": "abstract text",
            },
            status_code=200,
        )
        result = s2.get_paper("10.1109/TWC.2024.001")

    assert result["ok"] is True
    assert result["data"]["in_corpus"] == "smithDoiMatched2024"
    assert "paperId" not in result["data"]


def test_get_paper_upstream_error_is_structured():
    with requests_mock.Mocker() as m:
        m.get(_paper_url("DOI:10.1/nope"), status_code=404, text="not found")
        result = s2.get_paper("10.1/nope")

    assert result["ok"] is False
    assert "404" in result["error"]
