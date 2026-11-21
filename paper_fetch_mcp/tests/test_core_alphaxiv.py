"""Tests for paper_fetch_mcp.core.alphaxiv: paper id parsing and lookup.

All HTTP access goes through `_alphaxiv_fetch` (a thin urllib wrapper), which
is mocked directly -- no real network calls.
"""

from unittest.mock import patch

from paper_fetch_mcp.core import alphaxiv


# --- parse_paper_id -----------------------------------------------------

def test_parse_bare_id():
    assert alphaxiv.parse_paper_id("2401.12345") == "2401.12345"


def test_parse_versioned_id():
    assert alphaxiv.parse_paper_id("2401.12345v2") == "2401.12345v2"


def test_parse_arxiv_abs_url():
    assert alphaxiv.parse_paper_id("https://arxiv.org/abs/2401.12345") == "2401.12345"


def test_parse_arxiv_abs_url_versioned():
    assert alphaxiv.parse_paper_id("https://arxiv.org/abs/2401.12345v3") == "2401.12345v3"


def test_parse_arxiv_pdf_url():
    assert alphaxiv.parse_paper_id("https://arxiv.org/pdf/2401.12345") == "2401.12345"


def test_parse_alphaxiv_overview_url():
    assert alphaxiv.parse_paper_id("https://alphaxiv.org/overview/2401.12345") == "2401.12345"


def test_parse_alphaxiv_abs_url():
    assert alphaxiv.parse_paper_id("https://alphaxiv.org/abs/2401.12345") == "2401.12345"


def test_parse_unrecognised_input_returns_none():
    assert alphaxiv.parse_paper_id("not a paper id at all") is None


def test_parse_strips_whitespace():
    assert alphaxiv.parse_paper_id("  2401.12345  ") == "2401.12345"


# --- fetch ---------------------------------------------------------------

def test_fetch_returns_body_and_status_on_success():
    class _Resp:
        status = 200
        def read(self):
            return b"hello"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", return_value=_Resp()):
        body, status = alphaxiv.fetch("https://alphaxiv.org/overview/2401.12345.md")
    assert body == "hello"
    assert status == 200


def test_fetch_http_error_returns_empty_body_and_code():
    import urllib.error

    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError("url", 404, "not found", None, None),
    ):
        body, status = alphaxiv.fetch("https://alphaxiv.org/overview/9999.99999.md")
    assert body == ""
    assert status == 404


def test_fetch_network_error_returns_status_zero():
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        body, status = alphaxiv.fetch("https://alphaxiv.org/overview/2401.12345.md")
    assert status == 0
    assert "network down" in body.lower() or "Network error" in body


# --- lookup ----------------------------------------------------------------

def test_lookup_unparseable_id_returns_error_without_any_fetch():
    with patch.object(alphaxiv, "_alphaxiv_fetch") as mock_fetch:
        result = alphaxiv.lookup("this is definitely not a paper id")
    mock_fetch.assert_not_called()
    assert result["status"] == "error"
    assert "Could not parse" in result["error"]


def test_lookup_overview_success():
    with patch.object(alphaxiv, "_alphaxiv_fetch", return_value=("# Overview text", 200)) as mock_fetch:
        result = alphaxiv.lookup("2401.12345")

    assert result["status"] == "ok"
    assert result["source"] == "alphaxiv_overview"
    assert result["overview"] == "# Overview text"
    assert result["paper_id"] == "2401.12345"
    # Only the overview URL should have been fetched.
    assert mock_fetch.call_count == 1
    assert "overview" in mock_fetch.call_args[0][0]


def test_lookup_falls_back_to_full_text_when_overview_404s():
    def fake_fetch(url):
        if "overview" in url:
            return ("", 404)
        return ("# Full text of the paper", 200)

    with patch.object(alphaxiv, "_alphaxiv_fetch", side_effect=fake_fetch) as mock_fetch:
        result = alphaxiv.lookup("2401.12345")

    assert result["status"] == "ok"
    assert result["source"] == "alphaxiv_fulltext"
    assert result["full_text"] == "# Full text of the paper"
    assert mock_fetch.call_count == 2  # overview (404) then full text (200)


def test_lookup_both_endpoints_404_returns_not_found_with_pdf_hint():
    with patch.object(alphaxiv, "_alphaxiv_fetch", return_value=("", 404)):
        result = alphaxiv.lookup("2401.12345")

    assert result["status"] == "not_found"
    assert "arxiv.org/pdf/2401.12345" in result["error"]


def test_lookup_non_404_non_200_returns_error():
    with patch.object(alphaxiv, "_alphaxiv_fetch", return_value=("server exploded", 500)):
        result = alphaxiv.lookup("2401.12345")

    assert result["status"] == "error"
    assert result["error"] == "server exploded"


def test_lookup_truncates_overview_to_max_chars():
    long_text = "x" * 5000
    with patch.object(alphaxiv, "_alphaxiv_fetch", return_value=(long_text, 200)):
        result = alphaxiv.lookup("2401.12345", max_chars=100)

    assert len(result["overview"]) <= 100 + len("…[truncated]")
    assert result["overview"].endswith("…[truncated]")


def test_lookup_does_not_truncate_when_under_max_chars():
    short_text = "short overview"
    with patch.object(alphaxiv, "_alphaxiv_fetch", return_value=(short_text, 200)):
        result = alphaxiv.lookup("2401.12345", max_chars=15000)

    assert result["overview"] == short_text
