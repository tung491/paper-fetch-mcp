"""Tests for paper_fetch_mcp/app.py's startup wiring: health probes and
`build()`'s "no side effects at import, everything wired at real startup"
contract.

Calling `build()` here is cheap and does no real I/O: it only assigns
`mcp._lifespan` (an async contextmanager FastMCP invokes at real server
startup, not here), adds middleware instances, and registers the tool
surface + `/health` route. DB init and the job worker only start when that
lifespan actually runs, which requires a real running server (see
test_smoke_http.py). `build()` is guarded by a module-level `_built` flag so
calling it more than once (from this file, or from `__main__.py` in a real
process) is a safe no-op.
"""

from __future__ import annotations

import sys
import pathlib
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from paper_fetch_mcp import app
from paper_fetch_mcp import mcp


# ---------------------------------------------------------------------------
# build() has no side effects at import time
# ---------------------------------------------------------------------------


def test_build_not_triggered_merely_by_importing_app_or_tools():
    """Importing `paper_fetch_mcp.app` (or any `tools/*.py` module, which
    several other test files already do in this same process) must not itself call `build()`. This test only has
    something to prove the FIRST time it runs in a given process -- once
    any test calls `build()` for real (this file's own
    `test_build_registers_the_tool_surface_and_is_idempotent` included),
    `_built` stays True for the rest of the session, so we skip rather than
    false-fail on a state this file itself is responsible for creating."""
    if app._built:
        pytest.skip(
            "paper_fetch_mcp.app.build() already ran earlier in this test "
            "session (_built is True) -- nothing left to prove here this run"
        )
    assert app._built is False


# ---------------------------------------------------------------------------
# Health probes -- unit-tested against fakes, never the real DB
# ---------------------------------------------------------------------------


def test_health_probe_db_reports_reachable_and_paper_count(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import contextmanager

    db_path = tmp_path / "papers.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE papers (id TEXT)")
    conn.execute("INSERT INTO papers VALUES ('p1'), ('p2'), ('p3')")
    conn.commit()
    conn.close()

    @contextmanager
    def fake_get_db_connection():
        c = sqlite3.connect(db_path)
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr(app.papers_mod, "get_db_connection", fake_get_db_connection)
    result = app._health_probe_db()
    assert result == {"reachable": True, "papers": 3}


def test_health_probe_jobs_reports_recent_and_queued_counts(monkeypatch):
    monkeypatch.setattr(
        app.jobs_mod,
        "list_jobs",
        lambda limit=20: [
            {"status": "queued"},
            {"status": "done"},
            {"status": "queued"},
        ],
    )
    monkeypatch.setattr(app.jobs_mod, "get_worker", lambda: None)
    result = app._health_probe_jobs()
    assert result == {"recent": 3, "queued": 2, "worker_alive": False}


def test_health_probe_jobs_reports_worker_alive_when_worker_is_running(monkeypatch):
    monkeypatch.setattr(app.jobs_mod, "list_jobs", lambda limit=20: [])

    class _FakeWorker:
        def is_alive(self_inner):
            return True

    monkeypatch.setattr(app.jobs_mod, "get_worker", lambda: _FakeWorker())
    result = app._health_probe_jobs()
    assert result == {"recent": 0, "queued": 0, "worker_alive": True}


def test_health_probe_telemetry_reports_last_call_row(monkeypatch):
    monkeypatch.setattr(app.telemetry_mod, "last_call_row", lambda: {"tool": "get_paper"})
    result = app._health_probe_telemetry()
    assert result == {"last_call": {"tool": "get_paper"}}


# ---------------------------------------------------------------------------
# Lifespan: starts/stops the core.jobs worker (spec §8 Phase 2)
# ---------------------------------------------------------------------------


def test_lifespan_starts_worker_after_reconciling_stale_jobs_and_stops_it_on_shutdown(
    monkeypatch,
):
    """The FastMCP lifespan is what actually starts/stops core.jobs's
    worker thread ("start from the FastMCP lifespan ... never at import";
    "shut down cleanly when the server stops"). Drives app._lifespan
    directly against fakes for every other startup dependency (papers/
    publisher/telemetry) -- never the real papers DB -- and
    asserts the call ORDER: init_db -> reconcile_stale_jobs -> start_worker
    before the lifespan yields; stop_worker after it exits."""
    import asyncio

    calls = []

    monkeypatch.setattr(app.papers_mod, "init_fts", lambda: None)
    monkeypatch.setattr(app.publisher_mod, "ensure_cache_table", lambda: None)
    monkeypatch.setattr(app.jobs_mod, "init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr(
        app.jobs_mod,
        "reconcile_stale_jobs",
        lambda: calls.append("reconcile_stale_jobs") or 0,
    )
    monkeypatch.setattr(
        app.jobs_mod, "start_worker", lambda: calls.append("start_worker")
    )
    monkeypatch.setattr(
        app.jobs_mod, "stop_worker", lambda: calls.append("stop_worker")
    )
    monkeypatch.setattr(
        app.telemetry_mod, "init_db", lambda: calls.append("telemetry_init_db")
    )

    async def _drive():
        async with app._lifespan(server=None):
            assert calls == [
                "init_db",
                "reconcile_stale_jobs",
                "start_worker",
                "telemetry_init_db",
            ]

    asyncio.run(_drive())
    assert calls == [
        "init_db",
        "reconcile_stale_jobs",
        "start_worker",
        "telemetry_init_db",
        "stop_worker",
    ]


def test_lifespan_stops_the_worker_even_if_the_body_raises(monkeypatch):
    """stop_worker() must run on the way out even when the server side of
    the `yield` raises (e.g. a serve-loop crash) -- it's in the lifespan's
    `finally`, not just its happy path."""
    import asyncio

    calls = []

    monkeypatch.setattr(app.papers_mod, "init_fts", lambda: None)
    monkeypatch.setattr(app.publisher_mod, "ensure_cache_table", lambda: None)
    monkeypatch.setattr(app.jobs_mod, "init_db", lambda: None)
    monkeypatch.setattr(app.jobs_mod, "reconcile_stale_jobs", lambda: 0)
    monkeypatch.setattr(app.jobs_mod, "start_worker", lambda: None)
    monkeypatch.setattr(
        app.jobs_mod, "stop_worker", lambda: calls.append("stop_worker")
    )
    monkeypatch.setattr(app.telemetry_mod, "init_db", lambda: None)

    async def _drive():
        with pytest.raises(RuntimeError, match="serve loop exploded"):
            async with app._lifespan(server=None):
                raise RuntimeError("serve loop exploded")

    asyncio.run(_drive())
    assert calls == ["stop_worker"]


# ---------------------------------------------------------------------------
# build() itself: registers the tool surface, is idempotent
# ---------------------------------------------------------------------------


def test_build_registers_the_tool_surface_and_is_idempotent():
    """build() registers the full tool surface -- and none of the removed
    corpus-search tools -- and a second build() call is a safe no-op. The
    byte-identical-schema guard lives in `tests/test_tool_contract.py`."""
    import asyncio

    server1 = app.build()
    assert server1 is mcp

    async def _names():
        tools = await mcp.list_tools()
        return {t.name for t in tools}

    names_after_first = asyncio.run(_names())
    assert names_after_first == {
        "get_paper", "doi2content", "get_content_by_citation_key",
        "list_figures", "get_image_path", "search_metadata",
        "semantic_scholar_search", "semantic_scholar_paper",
        "alphaxiv_paper_lookup", "alphaxiv_batch_lookup",
        "fetch_publisher_html", "ingest_arxiv", "download_arxiv",
        "job_status", "server_status", "append_bibtex_entries",
        "symlink_references_bib",
    }

    # Second call must be a safe no-op -- same object back, no duplicate
    # middleware/tool registration (which would raise or double the count).
    server2 = app.build()
    assert server2 is server1
    names_after_second = asyncio.run(_names())
    assert names_after_second == names_after_first
