"""Tests for paper_fetch_mcp.core.papers (Wave 2b of the AgenticRAGServerV2
refactor). Fixtures adapted from tests/test_db.py and tests/test_content.py.

All tests build a small temp SQLite DB; nothing here touches the live
pre_processing_v2/papers.db.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary test database with sample data (mirrors
    tests/test_db.py's fixture)."""
    db_path = tmp_path / "test_papers.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
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
    c.execute(
        """
        INSERT INTO papers VALUES (
            'paper1', 'Deep Learning for SemCom', 'IEEE', 'TWC', 2024,
            'paper1', 'Smith and Jones', '10.1109/TWC.2024.001',
            'Smith2024Deep', 'This paper proposes a deep learning approach...', '[]'
        )
    """
    )
    c.execute(
        """
        INSERT INTO papers VALUES (
            'paper2', 'Transformer-Based Joint Coding', 'IEEE', 'TCOM', 2023,
            'paper2', 'Wang et al.', '10.1109/TCOM.2023.002',
            'Wang2023Transformer', 'We present a transformer architecture...', '[]'
        )
    """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def paper_setup(tmp_path):
    """Create temporary DB and content files for testing (mirrors
    tests/test_content.py's fixture)."""
    db_path = tmp_path / "papers.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY, title TEXT, publisher TEXT,
            booktitle_journal TEXT, year INTEGER, file TEXT,
            author TEXT, doi TEXT, citation_key TEXT,
            abstract TEXT, sections JSON
        )
    """
    )
    c.execute(
        """
        INSERT INTO papers VALUES (
            'testpaper', 'Test Paper', 'IEEE', 'TWC', 2024,
            'testpaper', 'Author', '10.1109/test',
            'Author2024Test', 'Test abstract text', '[]'
        )
    """
    )
    conn.commit()
    conn.close()

    mistral_dir = tmp_path / "mistral_output" / "testpaper" / "chunk"
    mistral_dir.mkdir(parents=True)
    (mistral_dir / "01_Introduction.md").write_text("# Introduction\nTest intro content.")
    (mistral_dir / "02_Proposed_System.md").write_text(
        "# Proposed System\nTest system content."
    )
    (mistral_dir / "03_Experiment.md").write_text("# Experiment\nTest experiment.")
    (mistral_dir / "04_Conclusion.md").write_text("# Conclusion\nTest conclusion.")

    return {
        "db_path": db_path,
        "mistral_dir": tmp_path / "mistral_output",
    }


# --------------------------------------------------------------------------
# Fuzzy resolve fix (finding F5)
# --------------------------------------------------------------------------


def test_fuzzy_resolve_regression_missing_column(tmp_path):
    """Regression test for F5: the old code did
    ``SELECT id, file, abstract FROM papers`` in _fuzzy_resolve and then read
    row["citation_key"] -- an unselected column. sqlite3.Row raises
    IndexError for that, which the function's own broad except swallowed,
    so it always returned None on a non-empty table.

    This test builds a key that only a *fuzzy* match can resolve (duplicated
    year suffix), bypassing the exact-match paths in resolve_citation_to_id,
    and asserts _fuzzy_resolve actually returns a hit. Run this test against
    the old buggy SELECT (id, file, abstract only) and it fails with
    ``IndexError: No item with that key`` surfacing as a None return --
    i.e. the assertion below fails.
    """
    db_path = tmp_path / "fuzzy.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY, title TEXT, publisher TEXT,
            booktitle_journal TEXT, year INTEGER, file TEXT,
            author TEXT, doi TEXT, citation_key TEXT,
            abstract TEXT, sections JSON
        )
        """
    )
    c.execute(
        """
        INSERT INTO papers VALUES (
            'ziyu1', 'Some Title', 'IEEE', 'Milcom', 2025,
            'ziyu1', 'Ziyu', NULL, 'ziyuMilcom20252025',
            'Abstract text', '[]'
        )
        """
    )
    conn.commit()
    conn.close()

    with patch("paper_fetch_mcp.core.papers.DB_PATH", db_path):
        from paper_fetch_mcp.core.papers import _fuzzy_resolve

        result = _fuzzy_resolve("ziyuMilcom2025")
        assert result is not None, (
            "fuzzy resolve returned None -- the citation_key column is "
            "missing from the SELECT again (F5 regression)"
        )
        assert result["id"] == "ziyu1"
        assert result["citation_key"] == "ziyuMilcom20252025"


def test_fuzzy_resolve_directly_exercises_all_three_selects(tmp_path):
    """Drive _fuzzy_resolve through the normalized-match branch (main SELECT)
    as well as both LIKE-fallback SELECTs, to ensure citation_key survives
    projection in all three query sites."""
    db_path = tmp_path / "fuzzy2.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY, title TEXT, publisher TEXT,
            booktitle_journal TEXT, year INTEGER, file TEXT,
            author TEXT, doi TEXT, citation_key TEXT,
            abstract TEXT, sections JSON
        )
        """
    )
    # Row that will only ever be found via the LIKE-prefix fallback (no
    # normalized match possible because the stored key has an extra suffix
    # word, not just a duplicated year).
    c.execute(
        """
        INSERT INTO papers VALUES (
            'abbr1', 'Some Title', 'IEEE', 'TWC', 2022,
            'abbr1', 'Author', NULL, 'smith2022deepnetworks',
            'Abstract', '[]'
        )
        """
    )
    conn.commit()
    conn.close()

    with patch("paper_fetch_mcp.core.papers.DB_PATH", db_path):
        from paper_fetch_mcp.core.papers import _fuzzy_resolve

        # "smith2022" is a prefix of the stored citation_key -> exercises the
        # citation_key LIKE fallback SELECT.
        result = _fuzzy_resolve("smith2022")
        assert result is not None
        assert result["citation_key"] == "smith2022deepnetworks"


# --------------------------------------------------------------------------
# resolve_citation_to_id: exact-then-fuzzy order
# --------------------------------------------------------------------------


def test_resolve_citation_to_id_exact_match_before_fuzzy(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import resolve_citation_to_id

        result = resolve_citation_to_id("Smith2024Deep")
        assert result is not None
        assert result["id"] == "paper1"

        # Fallback to id lookup (exact, still before fuzzy).
        result = resolve_citation_to_id("paper2")
        assert result is not None
        assert result["id"] == "paper2"

        result = resolve_citation_to_id("Nonexistent2024")
        assert result is None


def test_resolve_citation_to_id_falls_back_to_fuzzy(tmp_path):
    """When neither exact citation_key nor exact id matches, resolution must
    fall through to _fuzzy_resolve (duplicated-year case)."""
    db_path = tmp_path / "fuzzy3.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY, title TEXT, publisher TEXT,
            booktitle_journal TEXT, year INTEGER, file TEXT,
            author TEXT, doi TEXT, citation_key TEXT,
            abstract TEXT, sections JSON
        )
        """
    )
    c.execute(
        """
        INSERT INTO papers VALUES (
            'ziyu1', 'Some Title', 'IEEE', 'Milcom', 2025,
            'ziyu1', 'Ziyu', NULL, 'ziyuMilcom20252025',
            'Abstract text', '[]'
        )
        """
    )
    conn.commit()
    conn.close()

    with patch("paper_fetch_mcp.core.papers.DB_PATH", db_path):
        from paper_fetch_mcp.core.papers import resolve_citation_to_id

        result = resolve_citation_to_id("ziyuMilcom2025")
        assert result is not None
        assert result["id"] == "ziyu1"


def test_resolve_citation_strips_whitespace(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import resolve_citation_to_id

        result = resolve_citation_to_id("  Smith2024Deep  ")
        assert result is not None
        assert result["id"] == "paper1"


def test_resolve_doi_to_id(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import resolve_doi_to_id

        result = resolve_doi_to_id("10.1109/TWC.2024.001")
        assert result is not None
        assert result["id"] == "paper1"

        result = resolve_doi_to_id("10.9999/nonexistent")
        assert result is None


# --------------------------------------------------------------------------
# get_paper_content: abstract-on-"*" semantics (tools/content.py's, not
# mcp_server_v2.py's)
# --------------------------------------------------------------------------


def test_get_paper_content_all_sections_includes_abstract(paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper_content

        content = get_paper_content("Author2024Test", ["*"])
        assert "Test abstract text" in content
        assert "Test intro content" in content
        assert "Test conclusion" in content


def test_get_paper_content_specific_section_excludes_abstract(paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper_content

        content = get_paper_content("Author2024Test", ["introduction"])
        assert "Test intro content" in content
        assert "Test abstract text" not in content
        assert "Test conclusion" not in content


def test_get_paper_content_abstract_only(paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper_content

        content = get_paper_content("Author2024Test", ["abstract"])
        assert "Test abstract text" in content
        assert "Test intro content" not in content


def test_get_paper_content_not_found(paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import get_paper_content

        result = get_paper_content("Nonexistent2024", ["*"])
        assert "not found" in result.lower()


def test_get_content_by_doi_uses_wildcard_sections(paper_setup):
    """get_content_by_doi delegates to get_paper_content(sections=["*"]) when
    a citation_key is present, so it must also include the abstract."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_content_by_doi

        content = get_content_by_doi("10.1109/test")
        assert "Test abstract text" in content
        assert "Test intro content" in content


def test_get_content_by_doi_not_found(paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import get_content_by_doi

        result = get_content_by_doi("10.9999/nonexistent")
        assert "not found" in result.lower()


# --------------------------------------------------------------------------
# search_papers_fts: degrades gracefully with no FTS table
# --------------------------------------------------------------------------


def test_search_papers_fts_fallback_no_fts_table(tmp_db):
    """No papers_fts table exists in tmp_db (mirrors the live DB today, per
    the task brief: sqlite_master has only cached_html and papers)."""
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_papers_fts

        results = search_papers_fts("Deep Learning")
        assert len(results) >= 1
        assert results[0]["citation_key"] == "Smith2024Deep"


def test_search_papers_fts_uses_fts_table_when_present(tmp_db):
    """When init_fts() has been run, the FTS path is used instead of LIKE."""
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import init_fts, search_papers_fts

        init_fts()
        results = search_papers_fts("transformer")
        assert len(results) >= 1
        assert any(r["citation_key"] == "Wang2023Transformer" for r in results)


def test_init_fts_is_idempotent(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import init_fts

        init_fts()
        init_fts()  # must not raise


# --------------------------------------------------------------------------
# search_by_author / search_by_year_range (unchanged behaviour, sanity check)
# --------------------------------------------------------------------------


def test_search_by_author(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_by_author

        results = search_by_author("Smith")
        assert len(results) == 1
        assert results[0]["citation_key"] == "Smith2024Deep"

        results = search_by_author("Nonexistent")
        assert len(results) == 0


def test_search_by_year_range(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_by_year_range

        results = search_by_year_range(2023, 2024)
        assert len(results) == 2

        results = search_by_year_range(2024, 2024)
        assert len(results) == 1

        results = search_by_year_range(2025, 2026)
        assert len(results) == 0


def test_get_db_connection_context_manager(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import get_db_connection

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            count = cursor.fetchone()[0]
            assert count == 2


# --------------------------------------------------------------------------
# search_metadata: dispatch across argument combinations
# --------------------------------------------------------------------------


def test_search_metadata_no_filters_returns_error(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata()
        assert "error" in result
        assert result["results"] == []
        assert result["count"] == 0


def test_search_metadata_query_only(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(query="Deep Learning")
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Smith2024Deep"
        assert result["query"] == "Deep Learning"


def test_search_metadata_author_only(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(author="Wang")
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Wang2023Transformer"


def test_search_metadata_year_range_only(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(year_from=2023, year_to=2024)
        assert result["count"] == 2

        result = search_metadata(year_from=2024, year_to=2024)
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Smith2024Deep"


def test_search_metadata_year_from_only(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(year_from=2024)
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Smith2024Deep"


def test_search_metadata_query_and_author_combined(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        # Matches paper1 by query (title) AND author.
        result = search_metadata(query="Deep Learning", author="Smith")
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Smith2024Deep"

        # Query matches paper1's title, but author filter excludes it.
        result = search_metadata(query="Deep Learning", author="Wang")
        assert result["count"] == 0


def test_search_metadata_author_and_year_combined(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(author="Wang", year_from=2023, year_to=2023)
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Wang2023Transformer"

        result = search_metadata(author="Wang", year_from=2024, year_to=2024)
        assert result["count"] == 0


def test_search_metadata_all_filters_combined(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(
            query="transformer", author="Wang", year_from=2020, year_to=2023
        )
        assert result["count"] == 1
        assert result["results"][0]["citation_key"] == "Wang2023Transformer"


def test_search_metadata_respects_limit(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import search_metadata

        result = search_metadata(year_from=2000, year_to=2100, limit=1)
        assert result["count"] == 1


# --------------------------------------------------------------------------
# Phase 2 (finding F8): DOI normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10.1109/TWC.2024.001", "10.1109/twc.2024.001"),
        ("https://doi.org/10.1109/TWC.2024.001", "10.1109/twc.2024.001"),
        ("http://dx.doi.org/10.1109/TWC.2024.001", "10.1109/twc.2024.001"),
        ("HTTPS://DOI.ORG/10.1109/TWC.2024.001", "10.1109/twc.2024.001"),
        ("doi:10.1109/TWC.2024.001", "10.1109/twc.2024.001"),
        ("DOI:10.1109/TWC.2024.001", "10.1109/twc.2024.001"),
        ("  10.1109/TWC.2024.001  ", "10.1109/twc.2024.001"),
        ("http://dx.doi.org/10.1109/twc.2024.001", "10.1109/twc.2024.001"),
    ],
)
def test_normalize_doi_real_world_forms(raw, expected):
    from paper_fetch_mcp.core.papers import normalize_doi

    assert normalize_doi(raw) == expected


def test_normalize_doi_empty_and_none_are_passthrough():
    from paper_fetch_mcp.core.papers import normalize_doi

    assert normalize_doi("") == ""
    assert normalize_doi(None) is None


@pytest.mark.parametrize(
    "doi_input",
    [
        "10.1109/TWC.2024.001",
        "https://doi.org/10.1109/TWC.2024.001",
        "http://dx.doi.org/10.1109/TWC.2024.001",
        "http://dx.doi.org/10.1109/twc.2024.001",
        "doi:10.1109/TWC.2024.001",
        "DOI:10.1109/twc.2024.001",
        "  10.1109/twc.2024.001  ",
    ],
)
def test_resolve_doi_to_id_normalizes_real_world_forms(tmp_db, doi_input):
    """The stored DOI is mixed-case ('10.1109/TWC.2024.001'); every one of
    these real-world input forms (doi.org URL, dx.doi.org URL, 'doi:'
    prefix, whitespace, lowercase) must still resolve to the same row --
    finding F8: doi2content missed 88% of the time in part because
    'https://doi.org/10.1109/X' and '10.1109/x' were treated as different
    strings."""
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import resolve_doi_to_id

        result = resolve_doi_to_id(doi_input)
        assert result is not None
        assert result["id"] == "paper1"


def test_resolve_doi_to_id_still_none_for_unknown_doi(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import resolve_doi_to_id

        assert resolve_doi_to_id("https://doi.org/10.9999/nonexistent") is None


# --------------------------------------------------------------------------
# Phase 2 (finding F8): miss -> suggestions + next actions
# --------------------------------------------------------------------------


def test_suggest_citation_keys_finds_fuzzy_and_title_matches(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import suggest_citation_keys

        # Fuzzy prefix match against "Smith2024Deep".
        suggestions = suggest_citation_keys("Smith2024")
        assert any(s["citation_key"] == "Smith2024Deep" for s in suggestions)

        # Title/FTS match ("Transformer-Based Joint Coding").
        suggestions = suggest_citation_keys("Transformer")
        assert any(s["citation_key"] == "Wang2023Transformer" for s in suggestions)


def test_suggest_citation_keys_empty_query_returns_empty(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import suggest_citation_keys

        assert suggest_citation_keys("") == []
        assert suggest_citation_keys("   ") == []


def test_not_found_envelope_shape_and_suggestions(tmp_db):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", tmp_db):
        from paper_fetch_mcp.core.papers import not_found_envelope

        env = not_found_envelope("Smith2024")
        assert env["found"] is False
        assert env["query"] == "Smith2024"
        assert any(s["citation_key"] == "Smith2024Deep" for s in env["suggestions"])
        assert isinstance(env["next"], list) and env["next"]


def test_not_found_envelope_next_actions_for_doi_and_arxiv_id():
    from paper_fetch_mcp.core.papers import not_found_envelope

    doi_env = not_found_envelope("https://doi.org/10.1234/unknown")
    assert any("fetch_publisher_html" in a for a in doi_env["next"])

    arxiv_env = not_found_envelope("arXiv:2301.12345")
    assert any("ingest_arxiv 2301.12345" in a for a in arxiv_env["next"])


def test_not_found_envelope_generic_next_actions_for_plain_text():
    from paper_fetch_mcp.core.papers import not_found_envelope

    env = not_found_envelope("zhang2026Towards")
    assert env["next"] == ["search_metadata", "semantic_scholar_search"]


# --------------------------------------------------------------------------
# get_paper / list_figures fixture: a paper with an arXiv-style id, a DOI,
# a citation key, a title, chunk content, and figures both directly under
# the paper directory (real corpus layout) and under a "figures"
# subdirectory (get_image_path's other candidate layout).
# --------------------------------------------------------------------------


@pytest.fixture
def umbrella_paper_setup(tmp_path):
    db_path = tmp_path / "papers.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE papers (
            id TEXT PRIMARY KEY, title TEXT, publisher TEXT,
            booktitle_journal TEXT, year INTEGER, file TEXT,
            author TEXT, doi TEXT, citation_key TEXT,
            abstract TEXT, sections JSON
        )
        """
    )
    c.execute(
        """
        INSERT INTO papers VALUES (
            '2301.12345', 'A Great Paper About Semantic Comm', 'arXiv', NULL, 2023,
            '2301.12345', 'Author X', '10.4321/test.2023.999',
            'AuthorX2023Great', 'Great abstract text.', '[]'
        )
        """
    )
    conn.commit()
    conn.close()

    paper_dir = tmp_path / "mistral_output" / "2301.12345"
    chunk_dir = paper_dir / "chunk"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "01_Introduction.md").write_text(
        "# Introduction\n" + ("Intro body. " * 2000)
    )
    (chunk_dir / "02_Conclusion.md").write_text("# Conclusion\nConclusion body.")

    # Figures directly under the paper dir (real corpus layout).
    (paper_dir / "chunk_0_img-0.jpg").write_bytes(b"fake-jpg-bytes")
    (paper_dir / "chunk_0_img-1.png").write_bytes(b"fake-png-bytes")
    # Figures under a "figures" subdirectory (get_image_path's other layout).
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir()
    (figures_dir / "fig-extra.jpg").write_bytes(b"fake-jpg-bytes-2")

    return {
        "db_path": db_path,
        "mistral_dir": tmp_path / "mistral_output",
    }


# --------------------------------------------------------------------------
# resolve_any_identifier / get_paper: all four identifier kinds
# --------------------------------------------------------------------------


def test_resolve_any_identifier_by_citation_key(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import resolve_any_identifier

        row = resolve_any_identifier("AuthorX2023Great")
        assert row is not None
        assert row["id"] == "2301.12345"
        assert row["citation_key"] == "AuthorX2023Great"


def test_resolve_any_identifier_by_doi(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import resolve_any_identifier

        row = resolve_any_identifier("https://doi.org/10.4321/TEST.2023.999")
        assert row is not None
        assert row["id"] == "2301.12345"


def test_resolve_any_identifier_by_arxiv_id(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import resolve_any_identifier

        # Bare id (already covered by the exact-id path) ...
        row = resolve_any_identifier("2301.12345")
        assert row is not None and row["id"] == "2301.12345"

        # ... and an "arXiv:...v2"-decorated form, which only the
        # extract-then-retry arXiv branch can resolve (no exact/fuzzy
        # citation-key or id match for the full decorated string).
        row = resolve_any_identifier("arXiv:2301.12345v2")
        assert row is not None
        assert row["id"] == "2301.12345"


def test_resolve_any_identifier_by_title(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import init_fts, resolve_any_identifier

        init_fts()
        row = resolve_any_identifier("A Great Paper About Semantic Comm")
        assert row is not None
        assert row["id"] == "2301.12345"


def test_resolve_any_identifier_miss_returns_none(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import resolve_any_identifier

        assert resolve_any_identifier("totally-unknown-xyz-987") is None
        assert resolve_any_identifier("") is None


@pytest.mark.parametrize(
    "identifier",
    [
        "AuthorX2023Great",
        "https://doi.org/10.4321/TEST.2023.999",
        "arXiv:2301.12345v2",
        "A Great Paper About Semantic Comm",
    ],
)
def test_get_paper_resolves_all_four_identifier_kinds(umbrella_paper_setup, identifier):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper, init_fts

        init_fts()
        result = get_paper(identifier)
        assert result["found"] is True
        assert result["id"] == "2301.12345"
        assert result["citation_key"] == "AuthorX2023Great"
        assert result["doi"] == "10.4321/test.2023.999"
        assert "Intro body." in result["content"]


def test_get_paper_miss_returns_not_found_envelope(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import get_paper

        result = get_paper("zhang2026Towards")
        assert result["found"] is False
        assert result["query"] == "zhang2026Towards"
        assert "suggestions" in result
        assert "next" in result


# --------------------------------------------------------------------------
# get_paper: max_chars_per_paper / offset paging
# --------------------------------------------------------------------------


def test_get_paper_paging_via_offset(umbrella_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper

        full = get_paper("AuthorX2023Great", max_chars_per_paper=1_000_000, offset=0)
        total_chars = full["paging"]["total_chars"]
        assert total_chars > 2000  # the fixture's intro body is long

        page_size = 500
        first = get_paper("AuthorX2023Great", max_chars_per_paper=page_size, offset=0)
        assert first["found"] is True
        assert len(first["content"]) == page_size
        assert first["paging"] == {
            "offset": 0,
            "returned_chars": page_size,
            "total_chars": total_chars,
            "truncated": True,
        }

        second = get_paper(
            "AuthorX2023Great", max_chars_per_paper=page_size, offset=page_size
        )
        assert second["content"] == full["content"][page_size : page_size * 2]

        # Concatenating pages reproduces the full content.
        assert first["content"] + second["content"] == full["content"][: page_size * 2]

        # A window that reaches the end is not marked truncated.
        last = get_paper(
            "AuthorX2023Great",
            max_chars_per_paper=total_chars,
            offset=total_chars - 10,
        )
        assert last["paging"]["truncated"] is False
        assert last["paging"]["returned_chars"] == 10


def test_get_paper_paging_key_is_not_named_meta(umbrella_paper_setup):
    """core.output.respond() unconditionally overwrites a top-level 'meta'
    key on the payload it wraps -- get_paper's paging info must live under
    a different key ('paging') or it would be silently clobbered once
    wrapped by the tools/content.py wrapper."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper

        result = get_paper("AuthorX2023Great")
        assert "paging" in result
        assert "meta" not in result


# --------------------------------------------------------------------------
# list_figures
# --------------------------------------------------------------------------


def test_list_figures_against_real_paper_directory_layout(umbrella_paper_setup):
    """Exercises both figure layouts: directly under the paper dir (the
    real corpus's own layout) and under a 'figures' subdirectory."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import list_figures

        result = list_figures("AuthorX2023Great")
        assert result["found"] is True
        assert result["citation_key"] == "AuthorX2023Great"
        assert result["id"] == "2301.12345"
        assert result["count"] == 3

        figure_ids = {f["figure_id"] for f in result["figures"]}
        assert figure_ids == {"chunk_0_img-0.jpg", "chunk_0_img-1.png", "fig-extra.jpg"}
        for fig in result["figures"]:
            assert Path(fig["path"]).exists()


def test_list_figures_resolves_by_bare_id_too(umbrella_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import list_figures

        result = list_figures("2301.12345")
        assert result["found"] is True
        assert result["count"] == 3


def test_list_figures_miss_returns_not_found_envelope(umbrella_paper_setup):
    with patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]):
        from paper_fetch_mcp.core.papers import list_figures

        result = list_figures("NoSuchKeyXYZ123")
        assert result["found"] is False
        assert result["query"] == "NoSuchKeyXYZ123"
        assert "suggestions" in result
        assert "next" in result


def test_list_figures_paper_with_no_figures_returns_empty_list(paper_setup):
    """paper_setup's 'testpaper' fixture has chunk .md files but no image
    files at all -- list_figures must return found:true with an empty
    figures list, not a miss."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import list_figures

        result = list_figures("Author2024Test")
        assert result["found"] is True
        assert result["figures"] == []
        assert result["count"] == 0


# --------------------------------------------------------------------------
# save_paper_content / get_paper(save_path=...): write the FULL paper to disk
# --------------------------------------------------------------------------


def test_save_paper_content_writes_explicit_absolute_file(tmp_path):
    from paper_fetch_mcp.core.papers import save_paper_content

    target = tmp_path / "out" / "paper.md"  # parent does not exist yet
    report = save_paper_content("full body", str(target), default_stem="ignored")

    assert report["path"] == str(target)
    assert report["chars"] == len("full body")
    assert report["bytes"] == len("full body".encode("utf-8"))
    assert report["overwrote"] is False
    assert target.read_text(encoding="utf-8") == "full body"


def test_save_paper_content_relative_path_lands_under_paper_save_dir(tmp_path):
    """A relative save_path must NOT be resolved against the server
    process's cwd (whatever launched the MCP server) but under
    PAPER_SAVE_DIR."""
    with patch("paper_fetch_mcp.core.papers.PAPER_SAVE_DIR", tmp_path / "saved"):
        from paper_fetch_mcp.core.papers import save_paper_content

        report = save_paper_content("body", "nested/smith2024.md")

    assert report["path"] == str(tmp_path / "saved" / "nested" / "smith2024.md")
    assert (tmp_path / "saved" / "nested" / "smith2024.md").read_text() == "body"


@pytest.mark.parametrize("save_path", ["papers/", "papers"])
def test_save_paper_content_directory_path_uses_default_stem(tmp_path, save_path):
    """A trailing separator OR a suffix-less path means "directory": the
    file inside is named from the default stem, slugified."""
    with patch("paper_fetch_mcp.core.papers.PAPER_SAVE_DIR", tmp_path / "saved"):
        from paper_fetch_mcp.core.papers import save_paper_content

        report = save_paper_content("body", save_path, default_stem="Author X/2023")

    expected = tmp_path / "saved" / "papers" / "Author_X_2023.md"
    assert report["path"] == str(expected)
    assert expected.read_text() == "body"


def test_save_paper_content_existing_directory_is_treated_as_one(tmp_path):
    from paper_fetch_mcp.core.papers import save_paper_content

    existing = tmp_path / "already.md"  # has a suffix, but IS a directory
    existing.mkdir()
    report = save_paper_content("body", str(existing), default_stem="key2023")

    assert report["path"] == str(existing / "key2023.md")


def test_save_paper_content_reports_overwrite(tmp_path):
    from paper_fetch_mcp.core.papers import save_paper_content

    target = tmp_path / "paper.md"
    target.write_text("old")
    report = save_paper_content("new body", str(target))

    assert report["overwrote"] is True
    assert target.read_text() == "new body"


def test_save_paper_content_failure_is_reported_not_raised(tmp_path):
    """A write that cannot happen must come back as `{path, error}` rather
    than raising -- the paper content itself is still worth returning."""
    from paper_fetch_mcp.core.papers import save_paper_content

    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    report = save_paper_content("body", str(blocker / "sub" / "paper.md"))

    assert "error" in report
    assert "chars" not in report


def test_save_paper_content_empty_path_is_an_error(tmp_path):
    from paper_fetch_mcp.core.papers import save_paper_content

    assert save_paper_content("body", "   ") == {"error": "save_path was empty"}


def test_get_paper_save_path_writes_full_content_not_the_window(
    umbrella_paper_setup, tmp_path
):
    """The saved file is the WHOLE paper even when the returned content is
    a small window -- that is the point of save_path."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper

        target = tmp_path / "great.md"
        result = get_paper(
            "AuthorX2023Great", max_chars_per_paper=100, save_path=str(target)
        )

    assert result["found"] is True
    assert len(result["content"]) == 100  # the tool result is still windowed
    saved = target.read_text(encoding="utf-8")
    assert len(saved) == result["paging"]["total_chars"] > 100
    assert saved.startswith(result["content"])
    assert result["saved"]["path"] == str(target)
    assert result["saved"]["chars"] == result["paging"]["total_chars"]


def test_get_paper_save_path_directory_names_file_after_citation_key(
    umbrella_paper_setup, tmp_path
):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
        patch("paper_fetch_mcp.core.papers.PAPER_SAVE_DIR", tmp_path / "saved"),
    ):
        from paper_fetch_mcp.core.papers import get_paper

        result = get_paper("10.4321/test.2023.999", save_path="corpus/")

    expected = tmp_path / "saved" / "corpus" / "AuthorX2023Great.md"
    assert result["saved"]["path"] == str(expected)
    assert expected.exists()


def test_get_paper_without_save_path_writes_nothing(umbrella_paper_setup, tmp_path):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
        patch("paper_fetch_mcp.core.papers.PAPER_SAVE_DIR", tmp_path / "saved"),
    ):
        from paper_fetch_mcp.core.papers import get_paper

        result = get_paper("AuthorX2023Great")

    assert "saved" not in result
    assert not (tmp_path / "saved").exists()


def test_get_paper_miss_does_not_save(umbrella_paper_setup, tmp_path):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", umbrella_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            umbrella_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import get_paper

        target = tmp_path / "nothing.md"
        result = get_paper("NoSuchPaperXYZ", save_path=str(target))

    assert result["found"] is False
    assert "saved" not in result
    assert not target.exists()
