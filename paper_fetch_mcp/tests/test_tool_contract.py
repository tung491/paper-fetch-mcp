"""Golden contract test for PaperFetchServer's live tool surface.

This file pins two things against paper_fetch_mcp/tests/golden/tools_list.json:

1. ``tools`` -- the live tool roster + full input JSON schemas (parameter
   names, types, defaults, required-ness), regenerated at TEST TIME from
   ``await mcp.list_tools()`` (``.parameters`` IS the input JSON Schema).
   This part is a real, live diff -- a change must reproduce this exact
   roster/schema, or update the golden file on purpose.

2. ``response_key_paths`` -- the response-key paths agents actually read out
   of each tool's YAML envelope. These are NOT called live; they were
   derived by reading the tool bodies and are pinned as documentation with a
   source citation per tool. A lightweight source-grep sanity check (below)
   guards against the underlying key name silently vanishing from the
   source.

Re-blessing the golden file
----------------------------
When a diff reported by this file is a DELIBERATE, REVIEWED change (not a
regression), rerun with UPDATE_TOOL_CONTRACT=1 to rewrite
paper_fetch_mcp/tests/golden/tools_list.json from the current live state:

    UPDATE_TOOL_CONTRACT=1 uv run pytest paper_fetch_mcp/tests/test_tool_contract.py -q

Then review the diff to that JSON file like any other reviewed change.

Importing the `tools/*.py` modules is cheap -- it only registers `@mcp.tool`
decorators; DB init happens inside `paper_fetch_mcp.app.build()`'s FastMCP
lifespan, at real server startup. The `mcp_mod` fixture below stays
module-scoped purely so every test in this file shares one import.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "tools_list.json"

# Source files the RESPONSE_KEY_PATHS citations below point into. The canary
# test below greps the union of these files' text, so a stale citation into a
# file that no longer contains the token fails loudly.
_PKG_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    _PKG_DIR / "core" / "s2.py",
    _PKG_DIR / "core" / "papers.py",
    _PKG_DIR / "tools" / "content.py",
    _PKG_DIR / "tools" / "external.py",
]

TOOL_MODULES = ["admin", "content", "external", "ingest", "search"]

EXPECTED_TOOL_NAMES = [
    # content
    "get_paper",
    "doi2content",
    "get_content_by_citation_key",
    "list_figures",
    "get_image_path",
    # search (papers DB metadata only)
    "search_metadata",
    # external
    "semantic_scholar_search",
    "semantic_scholar_paper",
    "alphaxiv_paper_lookup",
    "alphaxiv_batch_lookup",
    "fetch_publisher_html",
    # ingest
    "ingest_arxiv",
    "download_arxiv",
    "job_status",
    # admin
    "server_status",
    "append_bibtex_entries",
    "symlink_references_bib",
]

# Corpus search (FAISS/BM25 retrieval, query rewriting) is deliberately not
# part of this server; these names must never reappear on the surface.
REMOVED_SEARCH_TOOL_NAMES = [
    "hybrid_search",
    "hyde_query",
    "stepback_query",
    "list_available_tags",
    "reload_indices",
]

# Response-key paths agents actually read out of each tool's envelope.
# Derived by READING the tool bodies -- never called live.
RESPONSE_KEY_PATHS: Dict[str, Dict[str, Any]] = {
    "get_paper": {
        "paths": ["found", "content", "paging.truncated"],
        "source": (
            "core/papers.py `get_paper` returns {found, citation_key, id, "
            "doi, title, content, paging: {offset, returned_chars, "
            "total_chars, truncated}}; tools/content.py passes it to "
            "`output.respond(result)` (or core/online.py's envelope on a "
            "corpus miss, which carries the same found/content/paging keys)."
        ),
    },
    "semantic_scholar_search": {
        "paths": ["papers[].title"],
        "source": (
            "core/s2.py requests the title field from the Semantic Scholar "
            "API and returns {'ok': True, 'data': {'total': ..., 'papers': "
            "[...]}}; tools/external.py's semantic_scholar_search then does "
            "`output.respond({'total': ..., 'papers': data.get('papers', [])}, "
            "list_key='papers')`."
        ),
    },
}


@pytest.fixture(scope="module")
def mcp_mod():
    """A FRESH import of the package with the five `tools/*` modules
    registered on a FRESH `paper_fetch_mcp.mcp` singleton, so this file's
    snapshot reflects a clean import regardless of what other test files in
    the same pytest session have already imported."""
    for key in list(sys.modules.keys()):
        if key == "paper_fetch_mcp" or key.startswith("paper_fetch_mcp."):
            del sys.modules[key]
    pkg = importlib.import_module("paper_fetch_mcp")
    for name in TOOL_MODULES:
        importlib.import_module(f"paper_fetch_mcp.tools.{name}")
    return pkg


def _live_tools_snapshot(mcp_mod) -> Dict[str, Any]:
    """Regenerate the live tool roster + input JSON schemas from the running
    FastMCP instance via ``await mcp.list_tools()`` (fastmcp 3.1.1) -- never
    by parsing source text."""

    async def _collect():
        return await mcp_mod.mcp.list_tools()

    tools = asyncio.run(_collect())
    return {
        t.name: {
            "parameters": t.parameters,
            "output_schema": t.output_schema,
        }
        for t in tools
    }


def _build_snapshot(mcp_mod) -> Dict[str, Any]:
    return {
        "_meta": {
            "description": (
                "Golden contract snapshot for PaperFetchServer's live "
                "tool surface. 'tools' is regenerated "
                "live at test time from mcp.list_tools() and diffed "
                "against this file. 'response_key_paths' is documentation "
                "derived by reading tool bodies (not called live) -- see "
                "the 'source' citation on each entry."
            ),
            "reblessing": (
                "Set UPDATE_TOOL_CONTRACT=1 and rerun "
                "paper_fetch_mcp/tests/test_tool_contract.py to rewrite "
                "this file from the current live state. Any diff is a "
                "deliberate, reviewed change -- review it like code."
            ),
        },
        "tools": _live_tools_snapshot(mcp_mod),
        "response_key_paths": RESPONSE_KEY_PATHS,
    }


def _load_golden() -> Dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _write_golden(data: Dict[str, Any]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The contract test
# ---------------------------------------------------------------------------


def test_tool_roster_and_schemas_match_golden(mcp_mod):
    """Regenerate the live tool roster + schemas and diff against the golden
    file. On mismatch, fail with a readable per-tool diff.

    Re-bless a deliberate change with:
        UPDATE_TOOL_CONTRACT=1 uv run pytest paper_fetch_mcp/tests/test_tool_contract.py -q
    """
    live = _build_snapshot(mcp_mod)

    if os.environ.get("UPDATE_TOOL_CONTRACT") == "1":
        _write_golden(live)
        pytest.skip(
            "UPDATE_TOOL_CONTRACT=1: golden file rewritten from the live "
            "snapshot, not compared this run."
        )

    assert GOLDEN_PATH.exists(), (
        f"Golden file missing at {GOLDEN_PATH}. Generate it with "
        "UPDATE_TOOL_CONTRACT=1 uv run pytest "
        "paper_fetch_mcp/tests/test_tool_contract.py -q"
    )
    golden = _load_golden()

    live_tools = live["tools"]
    golden_tools = golden["tools"]
    live_names = set(live_tools)
    golden_names = set(golden_tools)

    missing = sorted(golden_names - live_names)
    added = sorted(live_names - golden_names)
    assert not missing and not added, (
        "Tool roster changed vs golden contract "
        f"({GOLDEN_PATH}).\n"
        f"  In golden but missing live: {missing}\n"
        f"  Live but not in golden: {added}\n"
        "If this is a deliberate, reviewed change, rerun with "
        "UPDATE_TOOL_CONTRACT=1 uv run pytest "
        "paper_fetch_mcp/tests/test_tool_contract.py -q to re-bless."
    )

    mismatches = []
    for name in sorted(live_names):
        if live_tools[name] != golden_tools[name]:
            mismatches.append(
                f"--- {name} (golden) ---\n"
                f"{json.dumps(golden_tools[name], indent=2, sort_keys=True)}\n"
                f"--- {name} (live) ---\n"
                f"{json.dumps(live_tools[name], indent=2, sort_keys=True)}\n"
            )

    assert not mismatches, (
        f"Tool schema(s) changed vs golden contract ({GOLDEN_PATH}):\n\n"
        + "\n".join(mismatches)
        + "\nIf this is a deliberate, reviewed change, rerun with "
        "UPDATE_TOOL_CONTRACT=1 uv run pytest "
        "paper_fetch_mcp/tests/test_tool_contract.py -q to re-bless the "
        "golden file, then review the diff to that file like any other "
        "code change."
    )


def test_live_roster_is_exactly_the_expected_tools(mcp_mod):
    live = _live_tools_snapshot(mcp_mod)
    assert sorted(live.keys()) == sorted(EXPECTED_TOOL_NAMES)


def test_corpus_search_tools_are_not_registered(mcp_mod):
    live = _live_tools_snapshot(mcp_mod)
    assert not set(REMOVED_SEARCH_TOOL_NAMES) & set(live.keys())


def test_no_corpus_tags_resource_is_registered(mcp_mod):
    async def _uris():
        return {str(r.uri) for r in await mcp_mod.mcp.list_resources()}

    assert "corpus://tags" not in asyncio.run(_uris())


def test_response_key_paths_match_golden_file():
    """This part of the golden file is hand-maintained documentation, not a
    live regeneration -- guard against the in-file RESPONSE_KEY_PATHS table
    and the golden JSON drifting apart from each other."""
    golden = _load_golden()
    assert golden["response_key_paths"] == RESPONSE_KEY_PATHS


def test_documented_response_key_tokens_still_present_in_source():
    """Canary against a silent rename: since RESPONSE_KEY_PATHS is
    documentation (not regenerated live), confirm the literal tokens it
    cites still appear in the source files it cites (SOURCE_PATHS)."""
    combined = "".join(p.read_text(encoding="utf-8") for p in SOURCE_PATHS)
    for token in ("found", "content", "paging", "truncated", "papers", "title"):
        assert token in combined, (
            f"token {token!r} used in RESPONSE_KEY_PATHS no longer found in "
            f"any of {[p.name for p in SOURCE_PATHS]} -- update the citation "
            "and re-bless with UPDATE_TOOL_CONTRACT=1 if the rename was "
            "deliberate."
        )
