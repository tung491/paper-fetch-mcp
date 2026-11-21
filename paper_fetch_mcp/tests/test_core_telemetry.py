"""Tests for paper_fetch_mcp.core.telemetry: CallLogMiddleware + /health.

Formerly an equivalence suite that compared core.telemetry against the
original paper_fetch_mcp/telemetry.py (to prove Wave 1a's move was
behaviour-preserving). Now that telemetry.py has been deleted -- its job
done -- these assert directly against core.telemetry, absorbing every
assertion that used to live in the now-deleted tests/test_telemetry.py plus
this file's own private-helper and no-side-effects-at-import checks. No
assertion from either source file was dropped in the conversion.

No mocks for the middleware tests: these build a tiny real FastMCP server,
add the real middleware, and call it through fastmcp.Client's in-memory
transport (a FastMCP instance passed directly as the client transport) --
proof the middleware actually fires, not just that its methods are
individually callable.
"""

import asyncio
import json
import sqlite3
import types
from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from paper_fetch_mcp.core import telemetry as core_telemetry


def test_no_side_effects_at_import():
    # Hard constraint from the task brief: importing core.telemetry must not
    # open a sqlite3 connection (i.e. must not create/touch the DB) as a
    # module-level side effect. Force a fresh import with sqlite3.connect
    # patched to raise -- if the import itself ever called connect(), this
    # would blow up.
    import importlib
    import sys
    import sqlite3 as _sqlite3

    sys.modules.pop("paper_fetch_mcp.core.telemetry", None)

    original_connect = _sqlite3.connect

    def _boom(*args, **kwargs):
        raise AssertionError("sqlite3.connect() was called at import time")

    _sqlite3.connect = _boom
    try:
        importlib.import_module("paper_fetch_mcp.core.telemetry")
    finally:
        _sqlite3.connect = original_connect
        # Re-import cleanly so the rest of the suite sees the real module.
        sys.modules.pop("paper_fetch_mcp.core.telemetry", None)
        importlib.import_module("paper_fetch_mcp.core.telemetry")


def test_resolve_path():
    p = Path("/tmp/does-not-need-to-exist/sub/telemetry.db")
    assert core_telemetry._resolve_path(p) == p
    assert core_telemetry._resolve_path(str(p)) == p
    assert core_telemetry._resolve_path(None) == core_telemetry.TELEMETRY_DB_PATH


def _make_server(db_path):
    mcp = FastMCP("telemetry-test-server")

    @mcp.tool
    def echo(text: str) -> str:
        return f"echo: {text}"

    @mcp.tool
    def boom(text: str) -> str:
        raise ValueError("kaboom")

    mcp.add_middleware(core_telemetry.CallLogMiddleware(db_path=db_path))
    return mcp


def test_init_db_creates_table_and_indices(tmp_path):
    db_path = tmp_path / "telemetry.db"
    core_telemetry.init_db(path=db_path)

    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tool_calls)")}
        assert cols == {
            "id",
            "ts",
            "tool",
            "args_json",
            "duration_ms",
            "result_chars",
            "truncated",
            "error",
            "client_id",
        }
        index_names = {row[1] for row in conn.execute("PRAGMA index_list(tool_calls)")}
        indexed_cols = set()
        for name in index_names:
            indexed_cols.update(row[2] for row in conn.execute(f"PRAGMA index_info({name})"))
        assert {"ts", "tool"} <= indexed_cols

        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()

    # Calling init_db again must not raise (CREATE TABLE IF NOT EXISTS).
    core_telemetry.init_db(path=db_path)


def test_extract_tool_name():
    ctx = types.SimpleNamespace(message=types.SimpleNamespace(name="my_tool"), method=None)
    assert core_telemetry._extract_tool_name(ctx) == "my_tool"

    ctx_no_name = types.SimpleNamespace(message=types.SimpleNamespace(), method="fallback_method")
    assert core_telemetry._extract_tool_name(ctx_no_name) == "fallback_method"


def test_extract_args_json():
    ctx = types.SimpleNamespace(message=types.SimpleNamespace(arguments={"a": 1, "b": "x" * 5000}))
    expected = json.dumps({"a": 1, "b": "x" * 5000}, default=str)[:2000]
    assert core_telemetry._extract_args_json(ctx) == expected

    ctx_empty = types.SimpleNamespace(message=types.SimpleNamespace(arguments=None))
    assert core_telemetry._extract_args_json(ctx_empty) == "{}"

    ctx_broken = types.SimpleNamespace(message=object())
    assert core_telemetry._extract_args_json(ctx_broken) == "{}"


def test_extract_client_id():
    ctx = types.SimpleNamespace(fastmcp_context=types.SimpleNamespace(client_id="abc"))
    assert core_telemetry._extract_client_id(ctx) == "abc"

    ctx_none = types.SimpleNamespace(fastmcp_context=None)
    assert core_telemetry._extract_client_id(ctx_none) is None

    ctx_broken = object()
    assert core_telemetry._extract_client_id(ctx_broken) is None


def test_measure_result():
    block1 = types.SimpleNamespace(text="hello ")
    block2 = types.SimpleNamespace(text="meta:\n  truncated: true\n")
    result = types.SimpleNamespace(content=[block1, block2])
    assert core_telemetry._measure_result(result) == (31, True)

    empty_result = types.SimpleNamespace(content=[])
    assert core_telemetry._measure_result(empty_result) == (0, False)

    no_content_result = types.SimpleNamespace()
    assert core_telemetry._measure_result(no_content_result) == (0, False)


def test_rss_mb_is_positive_and_stable_across_calls():
    r1 = core_telemetry._rss_mb()
    r2 = core_telemetry._rss_mb()
    assert r1 > 0 and r2 > 0
    # Two calls moments apart on the same process should agree closely;
    # allow a small tolerance for scheduler/GC noise.
    assert abs(r1 - r2) <= 1.0


def test_call_log_middleware_records_successful_call(tmp_path):
    db_path = tmp_path / "telemetry.db"
    core_telemetry.init_db(path=db_path)
    mcp = _make_server(db_path)

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("echo", {"text": "hello"})

    result = asyncio.run(run())
    assert result.data == "echo: hello"

    rows = core_telemetry.recent_calls(db_path=db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["tool"] == "echo"
    assert row["error"] is None
    assert row["result_chars"] > 0
    assert row["duration_ms"] >= 0
    assert row["truncated"] == 0
    assert "hello" in row["args_json"]

    last = core_telemetry.last_call_row(db_path=db_path)
    assert last == row


def test_call_log_middleware_records_error_and_reraises(tmp_path):
    db_path = tmp_path / "telemetry.db"
    core_telemetry.init_db(path=db_path)
    mcp = _make_server(db_path)

    async def run():
        async with Client(mcp) as client:
            await client.call_tool("boom", {"text": "x"})

    with pytest.raises(Exception):
        asyncio.run(run())

    row = core_telemetry.last_call_row(db_path=db_path)
    assert row is not None
    assert row["tool"] == "boom"
    assert row["error"] is not None
    assert "kaboom" in row["error"]
    assert row["result_chars"] == 0


def test_telemetry_never_breaks_tool_call_when_db_unwritable(tmp_path):
    # Point the middleware at a path whose parent directory doesn't exist,
    # so every write fails. The tool call must still succeed.
    bad_db_path = tmp_path / "does" / "not" / "exist" / "telemetry.db"
    mcp = _make_server(bad_db_path)

    async def run():
        async with Client(mcp) as client:
            return await client.call_tool("echo", {"text": "still works"})

    result = asyncio.run(run())
    assert result.data == "echo: still works"


def test_recent_calls_limit_and_ordering(tmp_path):
    db_path = tmp_path / "telemetry.db"
    core_telemetry.init_db(path=db_path)
    mcp = _make_server(db_path)

    async def run():
        async with Client(mcp) as client:
            for i in range(5):
                await client.call_tool("echo", {"text": str(i)})

    asyncio.run(run())

    rows = core_telemetry.recent_calls(limit=3, db_path=db_path)
    assert len(rows) == 3
    # Newest first.
    assert '"4"' in rows[0]["args_json"]
    assert '"3"' in rows[1]["args_json"]
    assert '"2"' in rows[2]["args_json"]


def test_recent_calls_and_last_call_row_on_missing_db(tmp_path):
    missing = tmp_path / "nope.db"
    assert core_telemetry.recent_calls(db_path=missing) == []
    assert core_telemetry.last_call_row(db_path=missing) is None


def test_register_health_ok_and_degraded():
    starlette_testclient = pytest.importorskip("starlette.testclient")

    mcp = FastMCP("health-test-server")

    @mcp.tool
    def noop() -> str:
        return "ok"

    def good_probe():
        return {"n": 42}

    core_telemetry.register_health(mcp, probes={"good": good_probe})
    app = mcp.http_app()

    with starlette_testclient.TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert isinstance(body["pid"], int)
        assert body["uptime_s"] >= 0
        assert body["rss_mb"] > 0
        assert body["checks"] == {"good": {"n": 42}}


def test_register_health_degraded_when_probe_raises():
    starlette_testclient = pytest.importorskip("starlette.testclient")

    mcp = FastMCP("health-test-server-2")

    @mcp.tool
    def noop() -> str:
        return "ok"

    def bad_probe():
        raise RuntimeError("db unreachable")

    core_telemetry.register_health(mcp, probes={"db": bad_probe})
    app = mcp.http_app()

    with starlette_testclient.TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert "db unreachable" in body["checks"]["db"]["error"]


def test_register_health_with_no_probes():
    starlette_testclient = pytest.importorskip("starlette.testclient")

    mcp = FastMCP("health-test-server-3")

    @mcp.tool
    def noop() -> str:
        return "ok"

    core_telemetry.register_health(mcp)
    app = mcp.http_app()

    with starlette_testclient.TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"] == {}


def test_public_and_private_surface_present():
    # Every name the task brief calls out as "must be preserved" is present
    # on the moved module.
    for name in (
        "init_db",
        "recent_calls",
        "last_call_row",
        "CallLogMiddleware",
        "register_health",
        "_resolve_path",
        "_row_to_dict",
        "_extract_tool_name",
        "_extract_args_json",
        "_extract_client_id",
        "_measure_result",
        "_rss_mb",
        "TELEMETRY_DB_PATH",
    ):
        assert hasattr(core_telemetry, name), f"core.telemetry missing {name}"
