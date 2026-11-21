"""Tests for paper_fetch_mcp.tools.content's `@mcp.tool` wrappers (Phase 2 of
the AgenticRAGServerV2 refactor, spec §6/§8 "Phase 2", finding F8).

These call the tool functions directly (FastMCP's `@mcp.tool` decorator
returns the original plain function -- see `tests/test_server_v2_hotfix.py`'s
"retargeted to call the real tool functions directly" note) and parse the
returned YAML with `yaml.safe_load`. Every test patches
`paper_fetch_mcp.core.papers.DB_PATH`/`MISTRAL_OUTPUT_DIR` with a temp
SQLite DB + temp directory tree -- nothing here touches the live
pre_processing_v2/papers.db or its mistral_output.

NOTE: `paper_fetch_mcp.tools.content` is imported freshly INSIDE each test,
after the `patch(...)` context managers are entered -- never at this file's
top level. `tests/test_tool_contract.py`'s `mcp_mod` fixture deletes every
`paper_fetch_mcp.*` entry from `sys.modules` and reimports the package (by
design, to get a pristine tool roster snapshot); if that file's tests run
first in the same pytest session (alphabetically, `test_tool_contract.py` <
`test_tools_content.py`), a module-level `from paper_fetch_mcp.tools import
content` captured before that reload would keep calling the OLD `content`
module, whose own `papers` reference still points to the OLD `core.papers`
module object -- so `patch("paper_fetch_mcp.core.papers.DB_PATH", ...)`
(which patches whatever module is CURRENTLY in `sys.modules`) would silently
patch a different, orphaned object and every test here would fall through to
the real corpus DB. Importing `content` fresh inside the patched block, the
same way `tests/test_core_papers.py` imports `paper_fetch_mcp.core.papers`
functions inside its own `with patch(...)` blocks, avoids that.
"""

import sqlite3
from unittest.mock import patch

import pytest
import yaml


@pytest.fixture
def tool_paper_setup(tmp_path):
    """One resolvable paper (citation key + DOI + title + chunk content +
    figures), mirroring `tests/test_core_papers.py`'s `umbrella_paper_setup`
    fixture but scoped to this file so tool-level tests don't depend on
    another test module's fixture."""
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
            'toolpaper1', 'Tool Test Paper', 'IEEE', 'TWC', 2024,
            'toolpaper1', 'Author Y', '10.1109/TOOL.2024.001',
            'AuthorY2024Tool', 'Tool test abstract.', '[]'
        )
        """
    )
    conn.commit()
    conn.close()

    chunk_dir = tmp_path / "mistral_output" / "toolpaper1" / "chunk"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "01_Introduction.md").write_text("# Introduction\nTool intro body.")

    fig_dir = tmp_path / "mistral_output" / "toolpaper1"
    (fig_dir / "chunk_0_img-0.jpg").write_bytes(b"fake-jpg")

    return {
        "db_path": db_path,
        "mistral_dir": tmp_path / "mistral_output",
    }


# --------------------------------------------------------------------------
# doi2content: normalisation + not-found envelope, hit path unchanged
# --------------------------------------------------------------------------


def test_doi2content_hit_returns_plain_content_unchanged(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(content.doi2content("10.1109/TOOL.2024.001"))
        assert "Tool intro body" in data["content"]
        assert "Tool test abstract" in data["content"]
        # Hit path shape is unchanged: no "found" key added.
        assert "found" not in data


@pytest.mark.parametrize(
    "doi_form",
    [
        "https://doi.org/10.1109/TOOL.2024.001",
        "http://dx.doi.org/10.1109/tool.2024.001",
        "doi:10.1109/TOOL.2024.001",
        "  10.1109/tool.2024.001  ",
    ],
)
def test_doi2content_normalises_real_world_doi_forms(tool_paper_setup, doi_form):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(content.doi2content(doi_form))
        assert "Tool intro body" in data["content"]


def test_doi2content_miss_returns_structured_envelope_with_suggestions(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(content.doi2content("10.9999/does-not-exist"))
        assert data["found"] is False
        assert data["query"] == "10.9999/does-not-exist"
        assert "suggestions" in data
        assert "next" in data
        assert data["next"]  # non-empty: at least one follow-up action


# --------------------------------------------------------------------------
# get_content_by_citation_key: per-key hit unchanged, per-key miss structured
# --------------------------------------------------------------------------


def test_get_content_by_citation_key_hit_shape_unchanged(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(
            content.get_content_by_citation_key(["AuthorY2024Tool"])
        )
        assert isinstance(data["AuthorY2024Tool"], str)
        assert "Tool intro body" in data["AuthorY2024Tool"]


def test_get_content_by_citation_key_mixed_hit_and_miss(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(
            content.get_content_by_citation_key(
                ["AuthorY2024Tool", "zhang2026Towards"]
            )
        )
        assert isinstance(data["AuthorY2024Tool"], str)
        assert "Tool intro body" in data["AuthorY2024Tool"]

        miss = data["zhang2026Towards"]
        assert isinstance(miss, dict)
        assert miss["found"] is False
        assert miss["query"] == "zhang2026Towards"
        assert "suggestions" in miss
        assert "next" in miss


# --------------------------------------------------------------------------
# get_paper: umbrella tool, all identifier kinds + paging
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "identifier",
    [
        "AuthorY2024Tool",
        "https://doi.org/10.1109/TOOL.2024.001",
        "toolpaper1",
        "Tool Test Paper",
    ],
)
def test_get_paper_tool_resolves_all_identifier_kinds(tool_paper_setup, identifier):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.core.papers import init_fts
        from paper_fetch_mcp.tools import content

        init_fts()
        data = yaml.safe_load(content.get_paper(identifier))
        assert data["found"] is True
        assert data["id"] == "toolpaper1"
        assert data["citation_key"] == "AuthorY2024Tool"
        assert "Tool intro body" in data["content"]
        assert "paging" in data
        assert "meta" in data  # respond()'s own envelope meta


def test_get_paper_tool_miss(tool_paper_setup):
    """With the online fallback switched off (conftest sets
    GET_PAPER_ONLINE_FALLBACK=0 for every test) a corpus miss is the same
    structured envelope as before, plus an `online` block saying the
    network was NOT tried and why."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(content.get_paper("nonexistent-xyz-987"))
        assert data["found"] is False
        assert "suggestions" in data
        assert "next" in data
        assert data["online"] == {"attempted": False, "reason": "GET_PAPER_ONLINE_FALLBACK=0"}


def test_get_paper_tool_miss_falls_through_to_online_fetch(tool_paper_setup):
    """A corpus miss hands the identifier, the local miss envelope and the
    paging args to `core.online.get_paper_online`; its envelope is what the
    tool returns (wrapped in the usual respond() meta)."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        online_hit = {
            "found": True,
            "source": "online",
            "query": "10.1109/OTHER.2024.999",
            "kind": "doi",
            "doi": "10.1109/other.2024.999",
            "arxiv_id": None,
            "url": "https://ieeexplore.ieee.org/document/999",
            "publisher": "IEEE",
            "status": "ok",
            "route": "exit_node",
            "content": "Fetched via the lab exit node.",
            "paging": {"offset": 0, "returned_chars": 30, "total_chars": 30, "truncated": False},
        }
        with patch(
            "paper_fetch_mcp.tools.content.online.get_paper_online", return_value=online_hit
        ) as fallback:
            data = yaml.safe_load(
                content.get_paper("10.1109/OTHER.2024.999", max_chars_per_paper=500, offset=7)
            )

        assert fallback.call_count == 1
        args, kwargs = fallback.call_args
        assert args == ("10.1109/OTHER.2024.999",)
        assert kwargs["max_chars_per_paper"] == 500 and kwargs["offset"] == 7
        assert kwargs["local_miss"]["found"] is False
        assert kwargs["local_miss"]["query"] == "10.1109/OTHER.2024.999"
        assert data["found"] is True
        assert data["source"] == "online" and data["route"] == "exit_node"
        assert data["content"] == "Fetched via the lab exit node."
        assert "meta" in data


def test_get_paper_tool_local_hit_never_goes_online(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        with patch("paper_fetch_mcp.tools.content.online.get_paper_online") as fallback:
            data = yaml.safe_load(content.get_paper("AuthorY2024Tool"))
        fallback.assert_not_called()
        assert data["found"] is True and data["citation_key"] == "AuthorY2024Tool"
        assert "source" not in data


def test_get_paper_tool_paging_via_offset(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        full = yaml.safe_load(
            content.get_paper("AuthorY2024Tool", max_chars_per_paper=1_000_000)
        )
        total_chars = full["paging"]["total_chars"]

        page1 = yaml.safe_load(
            content.get_paper("AuthorY2024Tool", max_chars_per_paper=10, offset=0)
        )
        page2 = yaml.safe_load(
            content.get_paper("AuthorY2024Tool", max_chars_per_paper=10, offset=10)
        )
        assert page1["content"] + page2["content"] == full["content"][:20]
        assert page1["paging"]["total_chars"] == total_chars
        assert page1["paging"]["truncated"] is True


def test_get_paper_tool_save_path_writes_full_paper(tool_paper_setup, tmp_path):
    """`save_path` writes the WHOLE paper to disk while the tool result
    itself stays windowed by max_chars_per_paper."""
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        target = tmp_path / "tool.md"
        data = yaml.safe_load(
            content.get_paper(
                "AuthorY2024Tool", max_chars_per_paper=20, save_path=str(target)
            )
        )

    assert data["found"] is True
    assert len(data["content"]) == 20
    assert data["saved"]["path"] == str(target)
    assert data["saved"]["chars"] == data["paging"]["total_chars"]
    assert data["saved"]["overwrote"] is False
    saved = target.read_text(encoding="utf-8")
    assert len(saved) == data["paging"]["total_chars"]
    assert "Tool intro body" in saved


def test_get_paper_tool_save_path_reaches_the_online_fallback(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        with patch(
            "paper_fetch_mcp.tools.content.online.get_paper_online",
            return_value={"found": False, "query": "x"},
        ) as fallback:
            content.get_paper("10.1109/OTHER.2024.999", save_path="/tmp/out.md")

    assert fallback.call_args.kwargs["save_path"] == "/tmp/out.md"


def test_get_paper_tool_save_failure_still_returns_the_paper(
    tool_paper_setup, tmp_path
):
    """An unwritable save_path must not cost the caller the content: the
    envelope is a normal hit with `saved: {path, error}`."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory would have to be")
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(
            content.get_paper(
                "AuthorY2024Tool", save_path=str(blocker / "sub" / "paper.md")
            )
        )

    assert data["found"] is True and "Tool intro body" in data["content"]
    assert "error" in data["saved"]


# --------------------------------------------------------------------------
# list_figures
# --------------------------------------------------------------------------


def test_list_figures_tool_hit(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(content.list_figures("AuthorY2024Tool"))
        assert data["found"] is True
        assert data["id"] == "toolpaper1"
        assert data["count"] == 1
        assert data["figures"][0]["figure_id"] == "chunk_0_img-0.jpg"


def test_list_figures_tool_miss(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        data = yaml.safe_load(content.list_figures("NoSuchKeyXYZ123"))
        assert data["found"] is False
        assert "suggestions" in data
        assert "next" in data


# --------------------------------------------------------------------------
# get_image_path: unchanged legacy tool, still works after the papers.py
# resolution-ladder additions.
# --------------------------------------------------------------------------


def test_get_image_path_unchanged_after_phase_2(tool_paper_setup):
    with (
        patch("paper_fetch_mcp.core.papers.DB_PATH", tool_paper_setup["db_path"]),
        patch(
            "paper_fetch_mcp.core.papers.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ),
    ):
        from paper_fetch_mcp.tools import content

        with patch(
            "paper_fetch_mcp.tools.content.MISTRAL_OUTPUT_DIR",
            tool_paper_setup["mistral_dir"],
        ):
            data = yaml.safe_load(
                content.get_image_path("AuthorY2024Tool", "chunk_0_img-0.jpg")
            )
            assert data["path"].endswith("chunk_0_img-0.jpg")
