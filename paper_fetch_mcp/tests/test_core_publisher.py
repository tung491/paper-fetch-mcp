"""Tests for paper_fetch_mcp.core.publisher (moved from tools/publisher_fetch.py
per the refactor design spec §5). Adapted from tests/test_publisher_fetch.py's
suite for the new public API names (leading underscores dropped) and the
relocated `ensure_cache_table` (was db._ensure_cached_html_table).

core.publisher gets its DB connection from core.papers.get_db_connection,
not paper_fetch_mcp/db.py -- db.py's module import triggers a `CREATE TABLE`
as a side effect (see db.py:265), and core.publisher must never import it
(Wave 5b task 2). Cache tests patch "paper_fetch_mcp.core.papers.get_db_connection"
accordingly.

All HTTP is mocked (requests_mock); no real Chrome/Selenium browser is ever
launched -- _get_stealth_driver is always patched out.
"""

import json
import sys
import pathlib
import sqlite3
import requests
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def _make_papers_db(tmp_path, name="test.db"):
    """Create a temp DB with a papers table, return its path."""
    db_path = str(tmp_path / name)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE papers (id TEXT, doi TEXT, publisher TEXT, "
        "title TEXT, abstract TEXT, citation_key TEXT)"
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# ensure_cache_table (relocated cached_html DDL)
# ---------------------------------------------------------------------------

def test_ensure_cache_table_creates_table(tmp_path):
    """ensure_cache_table creates the cached_html table if it doesn't exist."""
    db_path = _make_papers_db(tmp_path)

    def make_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    with patch("paper_fetch_mcp.core.papers.get_db_connection", side_effect=make_conn):
        from paper_fetch_mcp.core.publisher import ensure_cache_table
        ensure_cache_table()

    conn2 = sqlite3.connect(db_path)
    rows = conn2.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cached_html'"
    ).fetchall()
    conn2.close()
    assert len(rows) == 1, "cached_html table should exist after ensure_cache_table()"


def test_ensure_cache_table_idempotent(tmp_path):
    """Calling ensure_cache_table twice must not raise."""
    db_path = _make_papers_db(tmp_path, "test_idempotent.db")

    def make_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    with patch("paper_fetch_mcp.core.papers.get_db_connection", side_effect=make_conn):
        from paper_fetch_mcp.core.publisher import ensure_cache_table
        ensure_cache_table()
        ensure_cache_table()  # second call -- must not raise

    conn2 = sqlite3.connect(db_path)
    rows = conn2.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cached_html'"
    ).fetchall()
    conn2.close()
    assert len(rows) == 1, "only one cached_html table should exist"


def test_ensure_cache_table_not_called_at_import():
    """Importing core.publisher must not touch the DB (no side effects at import)."""
    import sys as _sys
    for key in list(_sys.modules.keys()):
        if key == "paper_fetch_mcp.core.publisher":
            del _sys.modules[key]
    with patch("paper_fetch_mcp.core.papers.get_db_connection") as mock_conn:
        import paper_fetch_mcp.core.publisher  # noqa: F401
        mock_conn.assert_not_called()


def test_core_publisher_import_never_opens_a_sqlite_connection():
    """Regression test for Wave 5b task 2: core/publisher.py used to do
    `from .. import config, db`, and importing `paper_fetch_mcp.db` runs its
    module-level `_ensure_cached_html_table()` (db.py:265), which opens a
    real sqlite3 connection and issues a CREATE TABLE at import time --
    even before any tool call. core.publisher must not import `db` at all,
    so no sqlite3 connection is ever opened just by importing it.
    """
    import sys as _sys
    for key in list(_sys.modules.keys()):
        if key == "paper_fetch_mcp.core.publisher":
            del _sys.modules[key]

    with patch("sqlite3.connect") as mock_connect:
        import paper_fetch_mcp.core.publisher  # noqa: F401
        mock_connect.assert_not_called()

    # core.publisher itself must not import paper_fetch_mcp.db (that module
    # runs a CREATE TABLE at import time) -- check its own module namespace
    # rather than sys.modules globally, since another test in this same
    # process may have legitimately imported paper_fetch_mcp.db already.
    import paper_fetch_mcp.core.publisher as _pub
    assert not hasattr(_pub, "db"), (
        "core.publisher must not bind the paper_fetch_mcp.db module as `db`"
    )


# ---------------------------------------------------------------------------
# classify_identifier tests
# ---------------------------------------------------------------------------

def test_classify_url_direct_ieee():
    from paper_fetch_mcp.core.publisher import classify_identifier
    url, source = classify_identifier("https://ieeexplore.ieee.org/document/1234567")
    assert url == "https://ieeexplore.ieee.org/document/1234567"
    assert source == "IEEE"


def test_classify_url_direct_springer():
    from paper_fetch_mcp.core.publisher import classify_identifier
    url, source = classify_identifier("https://link.springer.com/article/10.1007/s00000-000-0")
    assert url == "https://link.springer.com/article/10.1007/s00000-000-0"
    assert source == "Springer"


def test_classify_url_direct_sciencedirect():
    from paper_fetch_mcp.core.publisher import classify_identifier
    url, source = classify_identifier(
        "https://www.sciencedirect.com/science/article/pii/S0000000000000000"
    )
    assert url == "https://www.sciencedirect.com/science/article/pii/S0000000000000000"
    assert source == "ScienceDirect"


def test_classify_url_other():
    from paper_fetch_mcp.core.publisher import classify_identifier
    url, source = classify_identifier("https://www.mdpi.com/1424-8220/24/1/1")
    assert source == "Other"


def test_classify_doi_via_db(requests_mock, tmp_path):
    """DOI found in DB -> publisher source from DB, URL from doi.org."""
    db_path = str(tmp_path / "test_doi.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE papers (id TEXT, doi TEXT, publisher TEXT, "
        "title TEXT, abstract TEXT, citation_key TEXT)"
    )
    conn.execute(
        "INSERT INTO papers VALUES ('p1', '10.1109/TWC.2024.001', 'IEEE', 'Test', 'Abs', 'k1')"
    )
    conn.commit()
    conn.close()

    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )

    def make_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    with patch("paper_fetch_mcp.core.papers.get_db_connection", side_effect=make_conn):
        from paper_fetch_mcp.core.publisher import classify_identifier
        url, source = classify_identifier("10.1109/TWC.2024.001")
        assert source == "IEEE"
        assert "ieeexplore" in url


def test_classify_doi_via_doi_org_redirect(requests_mock):
    """DOI not in DB -> resolve via doi.org redirect (requests, no redirects)."""
    requests_mock.get(
        "https://doi.org/10.1109/UNKNOWN.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/9999999"},
    )
    from paper_fetch_mcp.core.publisher import classify_identifier
    url, source = classify_identifier("10.1109/UNKNOWN.001")
    assert source == "IEEE"
    assert "ieeexplore" in url


def test_classify_semantic_scholar_paper_id(requests_mock):
    """paperId -> SS API -> DOI -> doi.org redirect."""
    requests_mock.get(
        "https://api.semanticscholar.org/graph/v1/paper/some-paperId",
        json={"externalIds": {"DOI": "10.1109/TWC.2024.001"}},
    )
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    from paper_fetch_mcp.core.publisher import classify_identifier
    url, source = classify_identifier("some-paperId")
    assert source == "IEEE"
    assert "ieeexplore" in url


def test_classify_unresolvable_paper_id_raises(requests_mock):
    """paperId that SS can't resolve to a DOI -> ValueError."""
    requests_mock.get(
        "https://api.semanticscholar.org/graph/v1/paper/unknown-id",
        status_code=404,
    )
    from paper_fetch_mcp.core.publisher import classify_identifier
    with pytest.raises(ValueError):
        classify_identifier("unknown-id")


# ---------------------------------------------------------------------------
# fetch_via_dev_browser tests (browser always mocked -- never launched for real)
# ---------------------------------------------------------------------------

def test_fetch_via_dev_browser_success():
    """Stealth driver returns page_source after navigation."""
    mock_driver = MagicMock()
    mock_driver.page_source = "<html><body>Paper content</body></html>"

    with patch("paper_fetch_mcp.core.publisher._get_stealth_driver", return_value=mock_driver):
        from paper_fetch_mcp.core.publisher import fetch_via_dev_browser
        html_content = fetch_via_dev_browser("https://example.com/paper")
        assert "Paper content" in html_content
        mock_driver.get.assert_called_once_with("https://example.com/paper")


def test_fetch_via_dev_browser_driver_startup_failure():
    """Chrome/selenium-stealth fails to start -> raises BrowserError."""
    with patch("paper_fetch_mcp.core.publisher._get_stealth_driver",
               side_effect=RuntimeError("Chrome not found")):
        from paper_fetch_mcp.core.publisher import fetch_via_dev_browser, BrowserError
        with pytest.raises(BrowserError) as exc:
            fetch_via_dev_browser("https://example.com/paper")
        assert "Chrome" in str(exc.value)


def test_fetch_via_dev_browser_page_load_failure():
    """Exception during get() -> raises BrowserError."""
    mock_driver = MagicMock()
    mock_driver.get.side_effect = RuntimeError("navigation refused")

    with patch("paper_fetch_mcp.core.publisher._get_stealth_driver", return_value=mock_driver):
        from paper_fetch_mcp.core.publisher import fetch_via_dev_browser, BrowserError
        with pytest.raises(BrowserError) as exc:
            fetch_via_dev_browser("https://example.com/paper")
        assert "navigation refused" in str(exc.value)


# ---------------------------------------------------------------------------
# extract_paper_content tests
# ---------------------------------------------------------------------------

def test_extract_paper_content_ieee_abstract():
    from paper_fetch_mcp.core.publisher import extract_paper_content
    html = (
        "<html><body><nav>menu</nav>"
        "<div class='abstract'>"
        "<p>This paper presents a novel approach to something important and long enough.</p>"
        "</div>"
        "<footer>copyright</footer></body></html>"
    )
    md = extract_paper_content(html, "IEEE")
    assert "novel approach" in md
    assert "menu" not in md


def test_extract_paper_content_fallback_largest_element():
    from paper_fetch_mcp.core.publisher import extract_paper_content
    html = (
        "<html><body>"
        "<div>short</div>"
        "<div>" + ("This is the largest text block on the page. " * 10) + "</div>"
        "</body></html>"
    )
    md = extract_paper_content(html, "Other")
    assert "largest text block" in md


# ---------------------------------------------------------------------------
# fetch_abstract tests
# ---------------------------------------------------------------------------

def test_fetch_abstract_success(requests_mock):
    """Fallback fetches abstract from Semantic Scholar Graph API."""
    requests_mock.get(
        "https://api.semanticscholar.org/graph/v1/paper/test-paperId",
        json={"abstract": "This is the paper abstract."},
    )
    from paper_fetch_mcp.core.publisher import fetch_abstract
    result = fetch_abstract("test-paperId")
    assert "This is the paper abstract." in result
    assert "<p>" in result  # wrapped in <p> tags


def test_fetch_abstract_no_abstract(requests_mock):
    """No abstract -> returns None."""
    requests_mock.get(
        "https://api.semanticscholar.org/graph/v1/paper/test-paperId",
        json={"abstract": None},
    )
    from paper_fetch_mcp.core.publisher import fetch_abstract
    result = fetch_abstract("test-paperId")
    assert result is None


# ---------------------------------------------------------------------------
# resolve_to_paper_id tests
# ---------------------------------------------------------------------------

def test_resolve_to_paper_id_bare_id(requests_mock):
    """Bare paperId -> returned directly from SS."""
    requests_mock.get(
        "https://api.semanticscholar.org/graph/v1/paper/bareId123",
        json={"paperId": "bareId123"},
    )
    from paper_fetch_mcp.core.publisher import resolve_to_paper_id
    result = resolve_to_paper_id("bareId123")
    assert result == "bareId123"


def test_resolve_to_paper_id_doi_prefix(requests_mock):
    """DOI (10.xxx) -> SS DOI:{doi} endpoint -> paperId."""
    requests_mock.get(
        "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TWC.2024.001",
        json={"paperId": "resolved-paper-id"},
    )
    from paper_fetch_mcp.core.publisher import resolve_to_paper_id
    result = resolve_to_paper_id("10.1109/TWC.2024.001")
    assert result == "resolved-paper-id"


# ---------------------------------------------------------------------------
# get_cache / upsert_cache / orchestrate_fetch tests
# ---------------------------------------------------------------------------

def test_get_cache_hit(tmp_path):
    """get_cache returns row when entry exists."""
    import datetime
    db_path = str(tmp_path / "test_cache.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE cached_html (id, url, source, html, fetched_at, error_count)"
    )
    conn.execute(
        "INSERT INTO cached_html VALUES (?, ?, ?, ?, ?, ?)",
        ["10.1109/TWC.2024.001", "https://ieeexplore.ieee.org/doc/1",
         "IEEE", "<html>cached</html>",
         datetime.datetime.now(datetime.timezone.utc).isoformat(), 0]
    )
    conn.commit()
    conn.close()

    def make_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    with patch("paper_fetch_mcp.core.papers.get_db_connection", side_effect=make_conn):
        from paper_fetch_mcp.core.publisher import get_cache
        row = get_cache("10.1109/TWC.2024.001")
        assert row is not None
        assert row["html"] == "<html>cached</html>"
        assert row["source"] == "IEEE"


def test_get_cache_miss():
    """get_cache returns None when entry does not exist."""
    def empty_conn():
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        return c
    with patch("paper_fetch_mcp.core.papers.get_db_connection", side_effect=empty_conn):
        from paper_fetch_mcp.core.publisher import get_cache
        row = get_cache("nonexistent")
        assert row is None


def test_upsert_cache_then_get_cache_round_trip(tmp_path):
    """upsert_cache followed by get_cache returns the stored row."""
    db_path = str(tmp_path / "test_roundtrip.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE cached_html (id TEXT PRIMARY KEY, url, source, html, "
        "fetched_at, error_count)"
    )
    conn.commit()
    conn.close()

    def make_conn():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        return c

    with patch("paper_fetch_mcp.core.papers.get_db_connection", side_effect=make_conn):
        from paper_fetch_mcp.core.publisher import get_cache, upsert_cache
        upsert_cache("id1", "https://example.com", "Other", "<p>hi</p>")
        row = get_cache("id1")
        assert row["url"] == "https://example.com"
        assert row["html"] == "<p>hi</p>"
        assert row["error_count"] == 0

        # Upsert again -- ON CONFLICT should update in place, not duplicate.
        upsert_cache("id1", "https://example.com/v2", "Other", "<p>bye</p>", error_count=2)
        row2 = get_cache("id1")
        assert row2["url"] == "https://example.com/v2"
        assert row2["html"] == "<p>bye</p>"
        assert row2["error_count"] == 2


def test_orchestrator_cache_hit():
    """When cache hit, dev-browser is NOT called."""
    with patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser:
        mock_cache.return_value = {
            "id": "10.1109/TWC.2024.001",
            "url": "https://ieeexplore.ieee.org/doc/1",
            "source": "IEEE",
            "html": "<html>cached</html>",
            "fetched_at": "2026-03-31T00:00:00+00:00",
            "error_count": 0,
        }
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
        mock_browser.assert_not_called()
        assert "cache_hit" in result
        assert "cached" in result


def test_orchestrator_force_refresh_calls_requests_not_browser():
    """force_refresh=True bypasses cache, calls fetch_via_requests (requests-
    first per spec §6/F9) -- Selenium is never touched by default."""
    with patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_requests_fetch, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content") as mock_extract, \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert:
        mock_requests_fetch.return_value = ("<html>fresh</html>", "direct")
        mock_extract.return_value = "fresh markdown " + ("x" * 200)
        mock_cache.return_value = {
            "id": "10.1109/TWC.2024.001",
            "url": "https://ieeexplore.ieee.org/doc/1",
            "source": "IEEE",
            "html": "<html>cached</html>",
            "fetched_at": "2026-03-31T00:00:00+00:00",
            "error_count": 0,
        }
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001", force_refresh=True)
        mock_cache.assert_not_called()  # get_cache not called when force_refresh=True
        mock_requests_fetch.assert_called_once()
        mock_browser.assert_not_called()
        assert "fresh markdown" in result


def test_orchestrator_requests_fetch_error_falls_back_to_abstract_without_browser(requests_mock):
    """RequestsFetchError -> (PUBLISHER_FETCH_BROWSER unset, so Selenium is
    never attempted) -> resolve_to_paper_id -> fetch_abstract -> fallback
    result with error_count=1. This is the F9 fix: Chrome must not be
    touched unless an operator opts in."""
    from paper_fetch_mcp.core.publisher import RequestsFetchError
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    with patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_requests_fetch, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser, \
         patch("paper_fetch_mcp.core.publisher.fetch_abstract") as mock_abs, \
         patch("paper_fetch_mcp.core.publisher.resolve_to_paper_id") as mock_resolve, \
         patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert, \
         patch.dict("os.environ", {}, clear=False):
        import os as _os
        _os.environ.pop("PUBLISHER_FETCH_BROWSER", None)
        mock_requests_fetch.side_effect = RequestsFetchError("connection refused")
        mock_resolve.return_value = "resolved-paper-id-123"
        mock_abs.return_value = "<p>Abstract fallback</p>"
        mock_cache.return_value = None

        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
        assert "fallback" in result
        assert "Abstract fallback" in result
        mock_browser.assert_not_called()
        mock_resolve.assert_called_once_with("10.1109/TWC.2024.001")
        # Verify error_count was incremented to 1 (no prior errors)
        mock_upsert.assert_called_once()
        _, kwargs = mock_upsert.call_args
        assert kwargs.get("error_count") == 1


def test_orchestrator_falls_back_to_browser_when_env_var_set_and_requests_fails(requests_mock):
    """With PUBLISHER_FETCH_BROWSER=1, a RequestsFetchError falls through to
    Selenium as a second attempt before the abstract fallback."""
    from paper_fetch_mcp.core.publisher import RequestsFetchError
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    with patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_requests_fetch, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content") as mock_extract, \
         patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert, \
         patch.dict("os.environ", {"PUBLISHER_FETCH_BROWSER": "1"}):
        mock_requests_fetch.side_effect = RequestsFetchError("connection refused")
        mock_browser.return_value = "<html>fresh</html>"
        mock_extract.return_value = "fresh markdown from browser " + ("x" * 200)
        mock_cache.return_value = None

        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
        mock_browser.assert_called_once()
        assert "fresh markdown from browser" in result


def test_orchestrator_low_content_requests_fetch_falls_through(requests_mock):
    """A requests fetch that 'succeeds' but extracts almost nothing (e.g. a
    paywall/consent page) is treated as not-good-enough and falls through to
    the next stage rather than returning a near-empty result."""
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    with patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_requests_fetch, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content") as mock_extract, \
         patch("paper_fetch_mcp.core.publisher.fetch_abstract") as mock_abs, \
         patch("paper_fetch_mcp.core.publisher.resolve_to_paper_id") as mock_resolve, \
         patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert, \
         patch.dict("os.environ", {}, clear=False):
        import os as _os
        _os.environ.pop("PUBLISHER_FETCH_BROWSER", None)
        mock_requests_fetch.return_value = ("<html>paywall</html>", "direct")
        mock_extract.return_value = "no"  # far below _MIN_EXTRACTED_CONTENT_CHARS
        mock_resolve.return_value = "resolved-id"
        mock_abs.return_value = "<p>Abstract fallback</p>"
        mock_cache.return_value = None

        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
        mock_browser.assert_not_called()
        assert "fallback" in result
        assert "Abstract fallback" in result
        # exit node off and headless browser off (conftest) -> the own-network
        # requests GET is the only attempt
        mock_requests_fetch.assert_called_once_with(
            "https://ieeexplore.ieee.org/document/123", direct_only=True
        )


def test_orchestrator_total_failure(requests_mock):
    """requests fetch fails, browser disabled, abstract fetch also fails ->
    error status, nothing cached."""
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    from paper_fetch_mcp.core.publisher import RequestsFetchError
    with patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_requests_fetch, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser, \
         patch("paper_fetch_mcp.core.publisher.fetch_abstract") as mock_abs, \
         patch("paper_fetch_mcp.core.publisher.resolve_to_paper_id") as mock_resolve, \
         patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert, \
         patch.dict("os.environ", {}, clear=False):
        import os as _os
        _os.environ.pop("PUBLISHER_FETCH_BROWSER", None)
        mock_requests_fetch.side_effect = RequestsFetchError("connection refused")
        mock_resolve.return_value = None  # SS can't resolve either
        mock_abs.return_value = None       # no abstract either
        mock_cache.return_value = None

        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
        assert "status: error" in result
        assert "error" in result
        mock_browser.assert_not_called()
        # upsert_cache should NOT be called on total failure
        mock_upsert.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_via_requests tests (requests-first fetch, finding F9)
# ---------------------------------------------------------------------------

def test_fetch_via_requests_success(requests_mock):
    from paper_fetch_mcp.core.publisher import fetch_via_requests
    requests_mock.get(
        "https://ieeexplore.ieee.org/document/123",
        text="<html><body>Paper content</body></html>",
        status_code=200,
    )
    html = fetch_via_requests("https://ieeexplore.ieee.org/document/123")
    assert "Paper content" in html


def test_fetch_via_requests_http_error_raises(requests_mock):
    from paper_fetch_mcp.core.publisher import fetch_via_requests, RequestsFetchError
    requests_mock.get(
        "https://ieeexplore.ieee.org/document/404",
        status_code=404,
    )
    with pytest.raises(RequestsFetchError):
        fetch_via_requests("https://ieeexplore.ieee.org/document/404")


def test_fetch_via_requests_connection_error_raises():
    from paper_fetch_mcp.core.publisher import fetch_via_requests, RequestsFetchError
    with patch(
        "paper_fetch_mcp.core.publisher.requests.get",
        side_effect=requests.exceptions.ConnectionError("refused"),
    ):
        with pytest.raises(RequestsFetchError):
            fetch_via_requests("https://example.com/paper")


def test_publisher_fetch_browser_enabled_reads_env_at_call_time():
    from paper_fetch_mcp.core.publisher import _publisher_fetch_browser_enabled
    with patch.dict("os.environ", {"PUBLISHER_FETCH_BROWSER": "1"}):
        assert _publisher_fetch_browser_enabled() is True
    with patch.dict("os.environ", {"PUBLISHER_FETCH_BROWSER": "0"}):
        assert _publisher_fetch_browser_enabled() is False
    with patch.dict("os.environ", {}, clear=False):
        import os as _os
        _os.environ.pop("PUBLISHER_FETCH_BROWSER", None)
        assert _publisher_fetch_browser_enabled() is False


def test_orchestrator_bad_identifier_returns_error_without_network():
    """Unresolvable identifier short-circuits before any cache/browser work."""
    with patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        with patch(
            "paper_fetch_mcp.core.publisher.classify_identifier",
            side_effect=ValueError("Could not resolve paperId 'bad-id'"),
        ):
            result = orchestrate_fetch("bad-id")
        assert "status: error" in result
        mock_cache.assert_not_called()
        mock_browser.assert_not_called()


# ---------------------------------------------------------------------------
# Exit-node route (core/exit_node.py) inside the publisher fetch layer
# ---------------------------------------------------------------------------

_EXIT_NODE_AVAILABLE = {
    "mode": "auto",
    "exit_node": "tung-lab-pc",
    "proxy_url": "socks5h://127.0.0.1:1056",
    "available": True,
    "reason": "stubbed",
}


def _resp(status, text="<html></html>"):
    from unittest.mock import MagicMock
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = text
    r.raise_for_status.side_effect = (
        requests.HTTPError(f"{status} error") if status >= 400 else None
    )
    return r


def test_fetch_via_requests_routed_direct_success_reports_direct_route():
    from paper_fetch_mcp.core.publisher import fetch_via_requests_routed
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=_resp(200, "<p>x</p>")) as g:
        html, route = fetch_via_requests_routed("https://ieeexplore.ieee.org/document/1")
    assert (html, route) == ("<p>x</p>", "direct")
    assert g.call_count == 1 and "proxies" not in g.call_args.kwargs


def test_fetch_via_requests_routed_retries_403_via_exit_node(monkeypatch):
    """A direct 403 (paywall) from this host's network is retried once
    through the exit-node proxy; the exit node's 200 is what comes back."""
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core.publisher import fetch_via_requests_routed
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_NAME", "tung-lab-pc")
    monkeypatch.setattr(config, "EXIT_NODE_PROXY_URL", "socks5h://127.0.0.1:1056")
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch(
             "paper_fetch_mcp.core.exit_node.requests.get",
             side_effect=[_resp(403), _resp(200, "<article>full text</article>")],
         ) as g:
        html, route = fetch_via_requests_routed("https://ieeexplore.ieee.org/document/1")
    assert html == "<article>full text</article>"
    assert route == "exit_node"
    assert g.call_args_list[1].kwargs["proxies"] == {
        "http": "socks5h://127.0.0.1:1056",
        "https": "socks5h://127.0.0.1:1056",
    }


def test_fetch_via_requests_routed_error_names_both_routes(monkeypatch):
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core.publisher import fetch_via_requests_routed, RequestsFetchError
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_NAME", "tung-lab-pc")
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.exit_node.requests.get", side_effect=[_resp(403), _resp(403)]):
        with pytest.raises(RequestsFetchError) as ei:
            fetch_via_requests_routed("https://ieeexplore.ieee.org/document/1")
    msg = str(ei.value)
    assert "direct: HTTP 403" in msg and "via exit node tung-lab-pc" in msg


def test_fetch_via_requests_routed_reports_why_exit_node_was_skipped():
    """Exit node off (conftest): a direct 403 surfaces as an error that says
    the exit node was not used, and why."""
    from paper_fetch_mcp.core.publisher import fetch_via_requests_routed, RequestsFetchError
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=_resp(403)) as g:
        with pytest.raises(RequestsFetchError) as ei:
            fetch_via_requests_routed("https://ieeexplore.ieee.org/document/1")
    assert g.call_count == 1
    assert "exit node not used: EXIT_NODE_MODE=off" in str(ei.value)


def test_fetch_via_requests_routed_force_exit_node_unavailable_is_fetch_error():
    from paper_fetch_mcp.core.publisher import fetch_via_requests_routed, RequestsFetchError
    with patch("paper_fetch_mcp.core.exit_node.requests.get") as g:
        with pytest.raises(RequestsFetchError, match="unavailable"):
            fetch_via_requests_routed("https://x/y", force_exit_node=True)
    g.assert_not_called()


def test_orchestrator_paywall_stub_retries_via_exit_node(requests_mock, monkeypatch):
    """A direct 200 whose extracted body is a paywall stub is retried through
    the exit node BEFORE any Selenium / abstract fallback; the full text from
    the exit node is cached and returned with route=exit_node."""
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_NAME", "tung-lab-pc")
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    full = "full article text " + ("y" * 300)
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_fetch, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_dev_browser") as mock_browser, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content") as mock_extract, \
         patch("paper_fetch_mcp.core.publisher.fetch_abstract") as mock_abs, \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert, \
         patch.dict("os.environ", {}, clear=False):
        import os as _os
        _os.environ.pop("PUBLISHER_FETCH_BROWSER", None)
        mock_fetch.side_effect = [("<html>paywall</html>", "direct"), ("<html>full</html>", "exit_node")]
        mock_extract.side_effect = ["no", full]

        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")

    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "ok"
    assert data["route"] == "exit_node"
    assert data["html"] == full
    assert mock_fetch.call_args_list[0].args == ("https://ieeexplore.ieee.org/document/123",)
    assert mock_fetch.call_args_list[1].kwargs == {"force_exit_node": True}
    mock_browser.assert_not_called()
    mock_abs.assert_not_called()
    mock_upsert.assert_called_once_with(
        "10.1109/TWC.2024.001", "https://ieeexplore.ieee.org/document/123", "IEEE", full
    )


def test_orchestrator_paywall_stub_on_both_routes_falls_back_with_reasons(requests_mock, monkeypatch):
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_NAME", "tung-lab-pc")
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_fetch, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content", return_value="no"), \
         patch("paper_fetch_mcp.core.publisher.fetch_abstract", return_value="<p>Abstract</p>"), \
         patch("paper_fetch_mcp.core.publisher.resolve_to_paper_id", return_value="pid"), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache"), \
         patch.dict("os.environ", {}, clear=False):
        import os as _os
        _os.environ.pop("PUBLISHER_FETCH_BROWSER", None)
        mock_fetch.side_effect = [("<html>paywall</html>", "direct"), ("<html>paywall</html>", "exit_node")]

        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")

    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "fallback"
    assert "direct requests fetch returned too little" in data["fallback_reason"]
    assert "via exit node tung-lab-pc returned too little" in data["fallback_reason"]
    assert mock_fetch.call_count == 2


def test_orchestrator_cache_hit_reports_route_cache():
    with patch("paper_fetch_mcp.core.publisher.get_cache") as mock_cache:
        mock_cache.return_value = {
            "id": "10.1109/TWC.2024.001",
            "url": "https://ieeexplore.ieee.org/doc/1",
            "source": "IEEE",
            "html": "<html>cached</html>",
            "fetched_at": "2026-03-31T00:00:00+00:00",
            "error_count": 0,
        }
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
    import yaml as _yaml
    assert _yaml.safe_load(result)["route"] == "cache"


def test_resolve_doi_url_goes_through_exit_node_get():
    """doi.org is resolved through the same routed GET, so a blocked doi.org
    also benefits from the exit node; redirects are NOT followed."""
    from paper_fetch_mcp.core import exit_node
    from paper_fetch_mcp.core.publisher import _resolve_doi_url
    fake = _resp(302)
    fake.headers = {"Location": "https://link.springer.com/article/10.1007/x"}
    fake.url = "https://doi.org/10.1007/x"  # instance attr, not on the Response spec
    with patch(
        "paper_fetch_mcp.core.publisher.exit_node.get",
        return_value=exit_node.RouteResult(response=fake),
    ) as routed_get:
        url, source = _resolve_doi_url("10.1007/x")
    assert (url, source) == ("https://link.springer.com/article/10.1007/x", "Springer")
    assert routed_get.call_args.args == ("https://doi.org/10.1007/x",)
    assert routed_get.call_args.kwargs["allow_redirects"] is False


def _fallback_row(error_count):
    return {
        "id": "10.1109/TWC.2024.001",
        "url": "https://ieeexplore.ieee.org/doc/1",
        "source": "fallback",
        "html": "<p>abstract only</p>",
        "fetched_at": "2026-09-14T00:00:00+00:00",
        "error_count": error_count,
    }


def test_orchestrator_cached_fallback_is_retried_when_exit_node_available(requests_mock, monkeypatch):
    """An abstract-only fallback cached while the exit node was down must not
    pin the paper to its abstract once the exit node works."""
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    full = "full article text " + ("z" * 300)
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=_fallback_row(1)), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed",
               side_effect=[("<html>stub</html>", "direct"),
                            ("<html>full</html>", "exit_node")]) as mock_fetch, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content",
               side_effect=["no", full]), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "ok" and data["route"] == "exit_node" and data["html"] == full
    assert mock_fetch.call_count == 2
    mock_upsert.assert_called_once()


def test_orchestrator_cached_fallback_served_when_exit_node_unavailable():
    """Exit node off (conftest): the cached fallback is a normal cache hit."""
    with patch("paper_fetch_mcp.core.publisher.get_cache", return_value=_fallback_row(1)), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_fetch:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "cache_hit" and data["route"] == "cache"
    mock_fetch.assert_not_called()


def test_orchestrator_cached_fallback_sticks_after_retry_budget(monkeypatch):
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core.publisher import _MAX_FALLBACK_RETRIES_VIA_EXIT_NODE
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher.get_cache",
               return_value=_fallback_row(_MAX_FALLBACK_RETRIES_VIA_EXIT_NODE)), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_fetch:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
    import yaml as _yaml
    assert _yaml.safe_load(result)["status"] == "cache_hit"
    mock_fetch.assert_not_called()


def test_orchestrator_cached_real_content_is_never_retried(monkeypatch):
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    row = {**_fallback_row(0), "source": "IEEE", "html": "<html>cached</html>"}
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=row), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_fetch:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")
    import yaml as _yaml
    assert _yaml.safe_load(result)["status"] == "cache_hit"
    mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# Own network first: headless browser before the exit node
# ---------------------------------------------------------------------------

def test_orchestrator_headless_browser_on_own_network_before_exit_node(requests_mock, monkeypatch):
    """A bot-challenge / stub page from the own-network GET is rendered in the
    headless browser on the SAME network; the exit node is never touched
    when that yields the article."""
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    full = "full article text " + ("h" * 300)
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_headless_browser",
               return_value="<html>rendered</html>") as mock_headless, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed",
               return_value=("", "direct")) as mock_fetch, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content", side_effect=["", full]), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")

    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "ok" and data["route"] == "direct" and data["html"] == full
    mock_fetch.assert_called_once_with("https://ieeexplore.ieee.org/document/123", direct_only=True)
    assert mock_headless.call_args.args == ("https://ieeexplore.ieee.org/document/123",)
    assert callable(mock_headless.call_args.kwargs["figure_urls"])
    mock_upsert.assert_called_once()


def test_orchestrator_headless_browser_failure_then_exit_node(requests_mock, monkeypatch):
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core.publisher import BrowserError, RequestsFetchError
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_NAME", "tung-lab-pc")
    requests_mock.get(
        "https://doi.org/10.1109/TWC.2024.001",
        status_code=302,
        headers={"Location": "https://ieeexplore.ieee.org/document/123"},
    )
    full = "full article text " + ("e" * 300)
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_headless_browser",
               side_effect=BrowserError("timed out")), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as mock_fetch, \
         patch("paper_fetch_mcp.core.publisher.extract_paper_content", return_value=full), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache"):
        mock_fetch.side_effect = [
            RequestsFetchError("requests fetch failed: direct: 403 error"),
            ("<html>full</html>", "exit_node"),
        ]
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch("10.1109/TWC.2024.001")

    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "ok" and data["route"] == "exit_node"
    assert mock_fetch.call_args_list[0].kwargs == {"direct_only": True}
    assert mock_fetch.call_args_list[1].kwargs == {"force_exit_node": True}


# Springer's interstitial for a non-browser client (trimmed): its only text,
# the "couldn't load" notice, extracts to 209 chars -- above
# _MIN_EXTRACTED_CONTENT_CHARS -- so length alone let it through.
_SPRINGER_CLIENT_CHALLENGE = """<!DOCTYPE html><html lang="en"><head>
<link href="/_fs-ch-1T1wmsGaOgGaSxcX/assets/styles.css" rel="stylesheet" />
<title>Client Challenge</title></head><body>
<div id="loading-error" role="alert" aria-live="polite">
A required part of this site couldn\u2019t load. This may be due to a browser
extension, network issues, or browser settings. Please check your
connection, disable any ad blockers, or try using a different browser.
</div>
<script>loadScript('/_fs-ch-1T1wmsGaOgGaSxcX/errors.js')</script>
</body></html>"""


def test_is_bot_challenge():
    from paper_fetch_mcp.core.publisher import is_bot_challenge
    assert is_bot_challenge(_SPRINGER_CLIENT_CHALLENGE)
    assert is_bot_challenge("<html><head><title>Just a moment...</title></head></html>")
    assert is_bot_challenge('<div id="challenge-container"></div>')
    assert not is_bot_challenge("")
    assert not is_bot_challenge(
        "<html><head><title>ImageNet Large Scale Visual Recognition Challenge</title>"
        "</head><body><div class=c-article-body>text</div></body></html>"
    )


def test_orchestrator_springer_client_challenge_goes_to_headless_browser(monkeypatch):
    """Springer serves a plain GET a "Client Challenge" page whose extracted
    notice clears the length floor; it must not be returned (or cached) as
    the paper -- the headless browser on the own network renders the real
    article instead."""
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    url = "https://link.springer.com/article/10.1007/s11263-015-0816-y"
    with patch("paper_fetch_mcp.core.exit_node.availability", return_value=_EXIT_NODE_AVAILABLE), \
         patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_headless_browser",
               return_value='<html><div class="c-article-body">' + "article text " * 40
               + "</div></html>"), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed",
               return_value=(_SPRINGER_CLIENT_CHALLENGE, "direct")) as mock_fetch, \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as mock_upsert:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        result = orchestrate_fetch(url)

    import yaml as _yaml
    data = _yaml.safe_load(result)
    assert data["status"] == "ok" and data["route"] == "direct"
    assert "article text" in data["html"] and "couldn" not in data["html"]
    mock_fetch.assert_called_once_with(url, direct_only=True)
    assert "couldn" not in mock_upsert.call_args.args[3]


def test_headless_browser_bin_disabled_and_override(monkeypatch):
    from paper_fetch_mcp.core import publisher
    # conftest disables it
    assert publisher._headless_browser_bin() is None
    monkeypatch.setenv("PUBLISHER_HEADLESS_BROWSER", "1")
    monkeypatch.setenv("PUBLISHER_HEADLESS_BROWSER_BIN", "my-chrome")
    with patch("paper_fetch_mcp.core.publisher.shutil.which", return_value="/opt/my-chrome") as w:
        assert publisher._headless_browser_bin() == "/opt/my-chrome"
    w.assert_called_once_with("my-chrome")


class _FakePage:
    """Stands in for _DevToolsPage: `states` are the successive page-state
    polls, then the DOM, then the in-page figure fetch result."""

    def __init__(self, states, html="<html><body>rendered</body></html>", figures=None):
        self.states = list(states)
        self.html = html
        self.figures = figures or {}
        self.calls = []
        self.closed = False

    def call(self, method, **params):
        self.calls.append((method, params))
        return {}

    def evaluate(self, expression, await_promise=False):
        if expression.startswith("JSON.stringify(["):
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if isinstance(state, Exception):
                raise state
            return json.dumps(state)
        if expression == "document.documentElement.outerHTML":
            return self.html
        self.calls.append(("figures", expression))
        return self.figures

    def close(self):
        self.closed = True


def _run_headless(page, figure_urls=None):
    from paper_fetch_mcp.core import publisher
    proc = MagicMock()
    with patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher._launch_headless_browser",
               return_value=(proc, "ws://127.0.0.1:1/devtools/page/x")), \
         patch("paper_fetch_mcp.core.publisher._DevToolsPage", return_value=page), \
         patch("paper_fetch_mcp.core.publisher._SETTLE_POLL_S", 0):
        html = publisher.fetch_via_headless_browser("https://x/doc/1", figure_urls=figure_urls)
    return html, proc


def test_fetch_via_headless_browser_waits_out_the_challenge_then_returns_dom():
    from paper_fetch_mcp.core.publisher import BrowserError
    page = _FakePage([
        ["complete", 2060, True],            # AWS WAF challenge page
        BrowserError("Runtime.evaluate failed: Execution context was destroyed"),
        ["loading", 500, False],
        ["complete", 9000, False],
        ["complete", 9500, False],           # still rendering
        ["complete", 9500, False],
        ["complete", 9500, False],           # stable 3 polls -> done
    ])
    html, proc = _run_headless(page)
    assert html == page.html
    assert page.calls[0] == ("Page.navigate", {"url": "https://x/doc/1"})
    assert page.states == [["complete", 9500, False]]  # every poll consumed
    assert page.closed
    proc.terminate.assert_called_once()


def test_fetch_via_headless_browser_caches_figures_fetched_in_page(tmp_path, monkeypatch):
    import base64
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core import images
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", tmp_path)
    gif = b"GIF89a-fake"
    page = _FakePage(
        [["complete", 10, False]],
        figures={"https://x/media/fig1.gif": ["image/gif", base64.b64encode(gif).decode()]},
    )
    seen = []

    def figure_urls(html):
        seen.append(html)
        return ["https://x/media/fig1.gif", "https://x/media/fig2.gif"]

    _run_headless(page, figure_urls)
    assert seen == [page.html]
    fetch_js = [c for c in page.calls if c[0] == "figures"][0][1]
    assert '"https://x/media/fig1.gif", "https://x/media/fig2.gif"' in fetch_js
    assert images.cached("https://x/media/fig1.gif").read_bytes() == gif
    assert images.cached("https://x/media/fig2.gif") is None  # not returned -> not cached


def test_fetch_via_headless_browser_skips_figures_already_cached(tmp_path, monkeypatch):
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core import images
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", tmp_path)
    images.store("https://x/media/fig1.gif", b"GIF89a", "image/gif")
    page = _FakePage([["complete", 10, False]])
    _run_headless(page, lambda html: ["https://x/media/fig1.gif"])
    assert not [c for c in page.calls if c[0] == "figures"]


def test_fetch_via_headless_browser_timeout_propagates_and_cleans_up():
    from paper_fetch_mcp.core.publisher import BrowserError
    page = _FakePage([BrowserError("headless browser timed out after 30s")])
    with pytest.raises(BrowserError, match="timed out"):
        _run_headless(page)
    assert page.closed


def test_fetch_via_headless_browser_unavailable_or_launch_failure():
    from paper_fetch_mcp.core import publisher
    with patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value=None):
        with pytest.raises(publisher.BrowserError, match="no headless browser"):
            publisher.fetch_via_headless_browser("https://x/doc/1")
    with patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher._launch_headless_browser",
               side_effect=publisher.BrowserError("headless browser exited 1")):
        with pytest.raises(publisher.BrowserError, match="exited 1"):
            publisher.fetch_via_headless_browser("https://x/doc/1")


# ---------------------------------------------------------------------------
# extract_paper_content: math, citation links, figures, IEEE layout
# ---------------------------------------------------------------------------

_IEEE_PAGE = """<html><body>
<header>IEEE Xplore <a href="javascript:void()">Sign In</a></header>
<h1 class="document-title">A Paper Title</h1>
<div class="abstract-desktop-div"><h2>Abstract:</h2><div class="abstract-text">We study x_1 routing.</div>
<div class="stats-document-abstract-doi">DOI: <a href="https://doi.org/10.1/x">10.1/x</a></div></div>
<div id="article">
<p>Prior work [<a ref-type="bibr" anchor="ref1" href="javascript:void()">1</a>] uses
<inline-formula class="inline-formula"><tex-math notation="LaTeX"><span class="MathJax_Preview"></span>
<span class="MathJax"><nobr>rendered glyphs</nobr></span>
<script type="math/tex">F_{B}=0.5</script></tex-math></inline-formula>
 as the baseline.</p>
<disp-formula class="display-formula"><tex-math notation="LaTeX"><div class="MathJax_Display">glyphs</div>
<script type="math/tex; mode=display">a_{1} * b_{2}</script></tex-math>
<span class="formula"><span class="link">View Source</span><span class="tex tex2jax_ignore">a_{1} * b_{2}</span></span></disp-formula>
<div class="figure" id="fig1"><div class="img-wrap"><a href="/mediastore/x/fig1-large.gif">
<img class="document-ft-image" src="/mediastore/x/fig1-small.gif" alt="Fig. 1. - Sample topologies used."/></a></div>
<div class="figcaption"><b class="title">Fig. 1. </b><fig><p>Sample topologies used.</p></fig></div></div>
<p>See <img src="/assets/img/icon.support.gif" width="24" height="20" alt="icon"/> the
<a href="#sec2">next section</a> and <a href="/xpl/other">another paper</a>.</p>
</div>
<footer>Copyright IEEE</footer>
</body></html>"""


def test_extract_ieee_page_math_links_and_figures():
    from paper_fetch_mcp.core.publisher import extract_paper_content
    md = extract_paper_content(_IEEE_PAGE, "IEEE", "https://ieeexplore.ieee.org/document/1")
    assert md.startswith("# A Paper Title")
    assert "We study x\\_1 routing." in md or "We study x_1 routing." in md
    # IEEE layout: header/footer/site chrome gone
    assert "Sign In" not in md and "Copyright IEEE" not in md
    # citation and in-page links -> plain text, no javascript:
    assert "javascript:" not in md
    assert "Prior work [1] uses $F_{B}=0.5$ as the baseline." in md
    assert "next section" in md and "(#sec2)" not in md
    assert "[another paper](https://ieeexplore.ieee.org/xpl/other)" in md
    # formulas: TeX kept verbatim (no markdown escaping), glyph spans and the
    # "View Source" copy dropped
    assert "$$\na_{1} * b_{2}\n$$" in md
    assert "rendered glyphs" not in md and "glyphs" not in md and "View Source" not in md
    # figure: full-size image, absolute URL, short alt; icons dropped
    assert "![Fig. 1](https://ieeexplore.ieee.org/mediastore/x/fig1-large.gif)" in md
    assert "icon.support" not in md
    assert "**Fig. 1.** Sample topologies used." in md


def test_extract_keeps_formula_elements_the_noise_filter_used_to_eat():
    """Regression: the tag-name noise regex matched "form" inside
    "inline-formula"/"disp-formula" and silently deleted every formula."""
    from paper_fetch_mcp.core.publisher import extract_paper_content
    html = ("<html><body><div class='c-article-body'><p>Let "
            "<inline-formula><tex-math>$x^2$</tex-math></inline-formula> be given, "
            "and some more text to be long enough for the extractor.</p>"
            "<disp-formula><tex-math>y = mx + b</tex-math></disp-formula>"
            "<form>search</form></div></body></html>")
    md = extract_paper_content(html, "Springer")
    assert "Let $x^2$ be given" in md
    assert "$$\ny = mx + b\n$$" in md
    assert "search" not in md


def test_extract_lazy_loaded_image_uses_data_src():
    from paper_fetch_mcp.core.publisher import extract_paper_content
    html = ("<html><body><article><p>Body text that is long enough to be the article.</p>"
            "<img src='data:image/gif;base64,R0lGOD' data-src='/fig/2.png' alt='Figure 2'/>"
            "</article></body></html>")
    md = extract_paper_content(html, "Other", "https://pub.example/a/1")
    assert "![Figure 2](https://pub.example/fig/2.png)" in md


# ---------------------------------------------------------------------------
# PDF-only pages (IEEE Early Access): fetch the PDF with the page's session
# ---------------------------------------------------------------------------

_IEEE_EA_PAGE = (
    '<html><head><script>xplGlobal.document.metadata={"isEarlyAccess":true,'
    '"pdfUrl":"/stamp/stamp.jsp?tp=&arnumber=11743914","title":"T"};</script></head>'
    '<body><h1 class="document-title">Early Paper</h1>'
    '<div class="abstract-desktop-div"><h2>Abstract:</h2>'
    "<div class='abstract-text'>" + ("An abstract sentence. " * 20) + "</div></div>"
    "</body></html>"
)
_EA_PDF = "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=11743914&ref="


def test_lacks_full_text_and_pdf_only_url():
    from paper_fetch_mcp.core.publisher import lacks_full_text, pdf_only_url
    assert lacks_full_text(_IEEE_EA_PAGE, "IEEE")
    assert pdf_only_url(_IEEE_EA_PAGE, "IEEE") == _EA_PDF
    full = _IEEE_EA_PAGE.replace("</body>", '<div id="article"><p>text</p></div></body>')
    assert not lacks_full_text(full, "IEEE") and pdf_only_url(full, "IEEE") is None
    assert not lacks_full_text(_IEEE_EA_PAGE, "Springer")


def test_fetch_via_headless_browser_fetches_pdf_in_page(tmp_path, monkeypatch):
    import base64
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core import images
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", tmp_path)
    page = _FakePage(
        [["complete", 10, False]],
        html=_IEEE_EA_PAGE,
        figures={_EA_PDF: ["application/pdf", base64.b64encode(b"%PDF-1.5").decode()]},
    )
    from paper_fetch_mcp.core.publisher import pdf_only_url
    from paper_fetch_mcp.core import publisher
    with patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher._launch_headless_browser", return_value=(MagicMock(), "ws://x")), \
         patch("paper_fetch_mcp.core.publisher._DevToolsPage", return_value=page), \
         patch("paper_fetch_mcp.core.publisher._SETTLE_POLL_S", 0):
        publisher.fetch_via_headless_browser(
            "https://ieeexplore.ieee.org/document/11743914/",
            pdf_url=lambda html: pdf_only_url(html, "IEEE"),
        )
    fetch_js = [c for c in page.calls if c[0] == "figures"][0][1]
    assert json.dumps(_EA_PDF) in fetch_js and '"application/pdf"' in fetch_js
    assert images.cached(_EA_PDF).read_bytes() == b"%PDF-1.5"


def _orchestrate_headless(html, *, pdf_bytes=None, tmp_path, monkeypatch, abstract=None):
    """orchestrate_fetch with the own-network GET returning a WAF stub and the
    headless browser returning `html` (and caching `pdf_bytes` as its PDF)."""
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core import images
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", tmp_path)

    def headless(url, figure_urls=None, pdf_url=None):
        if pdf_bytes is not None:
            images.store(pdf_url(html), pdf_bytes, "application/pdf")
        return html

    with patch("paper_fetch_mcp.core.publisher._headless_browser_bin", return_value="/usr/bin/chromium"), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_headless_browser", side_effect=headless), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed", return_value=("", "direct")), \
         patch("paper_fetch_mcp.core.publisher.pdf_to_text", return_value="PDF BODY TEXT " * 30), \
         patch("paper_fetch_mcp.core.publisher.resolve_to_paper_id", return_value="pid"), \
         patch("paper_fetch_mcp.core.publisher.fetch_abstract", return_value=abstract), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as upsert:
        from paper_fetch_mcp.core.publisher import orchestrate_fetch
        import yaml as _yaml
        return _yaml.safe_load(orchestrate_fetch("https://ieeexplore.ieee.org/document/11743914/")), upsert


def test_orchestrator_pdf_only_page_returns_page_plus_pdf_text(tmp_path, monkeypatch):
    data, upsert = _orchestrate_headless(
        _IEEE_EA_PAGE, pdf_bytes=b"%PDF-1.5", tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    assert data["status"] == "ok" and data["route"] == "direct"
    html = data["html"]
    assert html.startswith("# Early Paper")
    assert f"[Full-text PDF]({_EA_PDF})" in html
    assert "PDF BODY TEXT" in html
    upsert.assert_called_once()


def test_orchestrator_abstract_only_page_is_the_fallback_not_success(tmp_path, monkeypatch):
    """No access to the PDF either: the page's abstract is NOT returned as
    'ok' (the exit node / other stages must still get their turn) -- it ends
    up as the fallback, ahead of Semantic Scholar's abstract."""
    data, upsert = _orchestrate_headless(
        _IEEE_EA_PAGE, tmp_path=tmp_path, monkeypatch=monkeypatch, abstract="<p>S2 abstract</p>"
    )
    assert data["status"] == "fallback"
    assert data["html"].startswith("# Early Paper") and "S2 abstract" not in data["html"]
    assert "abstract only, no full text or PDF" in data["fallback_reason"]
    assert upsert.call_args.kwargs.get("error_count") == 1


def test_pdf_to_text_prefers_pdftotext_then_pypdf(tmp_path):
    from paper_fetch_mcp.core import publisher
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    done = MagicMock(returncode=0, stdout="page one\fpage two\f", stderr="")
    with patch("paper_fetch_mcp.core.publisher.shutil.which", return_value="/usr/bin/pdftotext"), \
         patch("paper_fetch_mcp.core.publisher.subprocess.run", return_value=done) as run:
        assert publisher.pdf_to_text(pdf) == "page one\n\npage two"
    assert run.call_args.args[0] == ["/usr/bin/pdftotext", "-enc", "UTF-8", str(pdf), "-"]
    reader = MagicMock(pages=[MagicMock(**{"extract_text.return_value": "A"}),
                              MagicMock(**{"extract_text.return_value": "B"})])
    with patch("paper_fetch_mcp.core.publisher.shutil.which", return_value=None), \
         patch("pypdf.PdfReader", return_value=reader):
        assert publisher.pdf_to_text(pdf) == "A\n\nB"


# ---------------------------------------------------------------------------
# arXiv: LaTeXML HTML full text, else the PDF -- never just the abstract page
# ---------------------------------------------------------------------------

_ARXIV_HTML = """<html><body><nav>arXiv nav</nav>
<article class="ltx_document"><h1 class="ltx_title_document">A Paper</h1>
<p>We set <math alttext="x_{i}^{2}" display="inline"><mi>x</mi></math> as in
<cite class="ltx_cite">[<a class="ltx_ref" href="#bib.bib1">1</a>]</cite> and more text to be long.</p>
<table class="ltx_equation ltx_eqn_table" id="S3.E1"><tbody><tr class="ltx_equation">
<td class="ltx_eqn_cell"><math alttext="a=b" display="block"><mi>a</mi></math></td>
<td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">(1)</span></td>
</tr></tbody></table>
<figure class="ltx_figure"><svg><foreignObject><img alt="Refer to caption" src="2601.00001v1/x1.png" width="400" height="200"/></foreignObject><text>overlay</text></svg>
<figcaption><span class="ltx_tag ltx_tag_figure">Fig. 1: </span>A caption.</figcaption></figure>
<section class="ltx_bibliography"><h2>References</h2><ul><li class="ltx_bibitem" id="bib.bib1">[1] A. Author, A Title.</li></ul></section>
</article></body></html>"""

_ARXIV_ABS = (
    '<html><head><meta name="citation_arxiv_id" content="2601.00001"/>'
    '<meta name="citation_pdf_url" content="https://arxiv.org/pdf/2601.00001"/></head>'
    '<body><h1 class="title">Title:A Paper</h1><div class="authors">Authors:A. Author</div>'
    '<blockquote class="abstract">Abstract: ' + ("An abstract sentence. " * 15) + "</blockquote>"
    "<div class='extra-services'>Bibliographic Explorer Toggle</div></body></html>"
)


def test_url_to_source_arxiv():
    from paper_fetch_mcp.core.publisher import _url_to_source
    assert _url_to_source("https://arxiv.org/abs/2601.00001") == "arXiv"


def test_extract_arxiv_latexml_math_equations_figures_and_references():
    from paper_fetch_mcp.core.publisher import extract_paper_content
    md = extract_paper_content(_ARXIV_HTML, "arXiv", "https://arxiv.org/html/2601.00001")
    assert "arXiv nav" not in md
    assert "We set $x_{i}^{2}$ as in [1] and more text" in " ".join(md.split())
    assert "$$\na=b \\tag{1}\n$$" in md and "| " not in md  # not a Markdown table
    assert "![Fig. 1](https://arxiv.org/html/2601.00001v1/x1.png)" in md
    assert "overlay" not in md
    assert "## References" in md and "A. Author, A Title." in md


def test_arxiv_abstract_page_lacks_full_text_and_names_its_pdf():
    from paper_fetch_mcp.core.publisher import lacks_full_text, pdf_only_url
    assert lacks_full_text(_ARXIV_ABS, "arXiv")
    assert pdf_only_url(_ARXIV_ABS, "arXiv") == "https://arxiv.org/pdf/2601.00001"
    assert not lacks_full_text(_ARXIV_HTML, "arXiv")


def _routed_resp(status, text="", url="", ctype="text/html", content=b""):
    r = MagicMock(spec=requests.Response)
    r.status_code, r.text, r.url, r.content = status, text, url, content
    r.headers = {"Content-Type": ctype}
    return MagicMock(response=r, route="direct")


def test_fetch_arxiv_full_text_prefers_html(tmp_path, monkeypatch):
    from paper_fetch_mcp.core import publisher
    html_url = "https://arxiv.org/html/2601.00001v1"
    with patch("paper_fetch_mcp.core.publisher.exit_node.get",
               return_value=_routed_resp(200, _ARXIV_HTML, html_url)) as get, \
         patch("paper_fetch_mcp.core.images.download") as dl:
        found, reasons = publisher.fetch_arxiv_full_text("https://arxiv.org/abs/2601.00001v2")
    assert get.call_args.args[0] == "https://arxiv.org/html/2601.00001"
    assert found["url"] == html_url and "![Fig. 1](https://arxiv.org/html/2601.00001v1/x1.png)" in found["content"]
    assert reasons == []
    dl.assert_not_called()


def test_fetch_arxiv_full_text_falls_back_to_pdf(tmp_path, monkeypatch):
    from paper_fetch_mcp import config
    from paper_fetch_mcp.core import publisher, images
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", tmp_path)
    pdf_url = "https://arxiv.org/pdf/2601.00001"
    responses = {
        "https://arxiv.org/html/2601.00001": _routed_resp(404, "Not Found"),
        pdf_url: _routed_resp(200, url=pdf_url, ctype="application/pdf", content=b"%PDF-1.5"),
        "https://arxiv.org/abs/2601.00001": _routed_resp(200, _ARXIV_ABS, "https://arxiv.org/abs/2601.00001"),
    }
    with patch("paper_fetch_mcp.core.publisher.exit_node.get", side_effect=lambda u, **kw: responses[u]), \
         patch("paper_fetch_mcp.core.images.exit_node.get", side_effect=lambda u, **kw: responses[u]), \
         patch("paper_fetch_mcp.core.publisher.pdf_to_text", return_value="PDF BODY"):
        found, reasons = publisher.fetch_arxiv_full_text("https://arxiv.org/abs/2601.00001")
    assert found["url"] == "https://arxiv.org/abs/2601.00001"
    content = found["content"]
    assert content.index("Title:A Paper") < content.index(f"[Full-text PDF]({pdf_url})") < content.index("PDF BODY")
    assert "Bibliographic Explorer" not in content
    assert images.cached(pdf_url).read_bytes() == b"%PDF-1.5"
    assert reasons == ["arXiv has no HTML full text (HTTP 404)"]


def test_orchestrator_arxiv_returns_full_text_or_falls_back_to_abstract(tmp_path, monkeypatch):
    from paper_fetch_mcp import config
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", tmp_path)
    import yaml as _yaml
    from paper_fetch_mcp.core.publisher import orchestrate_fetch
    with patch("paper_fetch_mcp.core.publisher.fetch_arxiv_full_text",
               return_value=({"content": "FULL " * 100, "url": "https://arxiv.org/html/2601.00001v1",
                              "route": "direct"}, [])), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache") as upsert, \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed") as generic:
        data = _yaml.safe_load(orchestrate_fetch("https://arxiv.org/abs/2601.00001"))
    assert data["status"] == "ok" and data["url"] == "https://arxiv.org/html/2601.00001v1"
    generic.assert_not_called()
    upsert.assert_called_once()

    # neither HTML nor PDF: the generic chain gets the abstract page, which is
    # only the fallback, with the arXiv reasons in fallback_reason
    with patch("paper_fetch_mcp.core.publisher.fetch_arxiv_full_text",
               return_value=(None, ["arXiv PDF https://arxiv.org/pdf/2601.00001 could not be downloaded"])), \
         patch("paper_fetch_mcp.core.publisher.fetch_via_requests_routed", return_value=(_ARXIV_ABS, "direct")), \
         patch("paper_fetch_mcp.core.publisher.get_cache", return_value=None), \
         patch("paper_fetch_mcp.core.publisher.upsert_cache"):
        data = _yaml.safe_load(orchestrate_fetch("https://arxiv.org/abs/2601.00001"))
    assert data["status"] == "fallback" and "Title:A Paper" in data["html"]
    assert "could not be downloaded" in data["fallback_reason"]
    assert "abstract only" in data["fallback_reason"]
