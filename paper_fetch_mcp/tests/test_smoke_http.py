"""Opt-in HTTP smoke test: boots a real PaperFetchServer subprocess and
calls a few tools over streamable-http.

Skipped at import time unless ``PAPER_FETCH_SMOKE=1`` is set:

    PAPER_FETCH_SMOKE=1 uv run pytest paper_fetch_mcp/tests/test_smoke_http.py -q -s

A plain ``pytest`` run must stay green and fast, so it must never boot a
server -- hence the guard. Every test in here is also marked
``@pytest.mark.smoke`` (registered in pyproject.toml).

Safety: this module never talks to an already-running server and never
touches ``run_server.sh``'s PID file. It starts its own server subprocess
bound to 127.0.0.1 on an ephemeral port it picks itself, waits for that
subprocess's own ``/health`` endpoint, and always terminates that exact
subprocess (never any other PID) in a session-scoped fixture's teardown.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pytest
import yaml

# ---------------------------------------------------------------------------
# Opt-in guard -- must run before anything that imports the heavy server
# package or spends real time. Keep this the very first thing pytest sees.
# ---------------------------------------------------------------------------

if os.environ.get("PAPER_FETCH_SMOKE") != "1":
    pytest.skip(
        "opt-in only: set PAPER_FETCH_SMOKE=1 to run this smoke test (it boots "
        "a real server subprocess and calls Semantic Scholar)",
        allow_module_level=True,
    )

from fastmcp import Client  # noqa: E402  (import after the opt-in guard)

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARD_CAP = 40_000  # no tool response may exceed this
SERVER_BOOT_TIMEOUT_S = 60.0
QUERY = "deep learning based channel estimation for wireless communication systems"


# ---------------------------------------------------------------------------
# Private server subprocess (session-scoped: one server serves every test)
# ---------------------------------------------------------------------------


class _PortConflict(RuntimeError):
    """The ephemeral port we picked was grabbed by someone else before our
    subprocess could bind it. Caller should retry with a fresh port."""


class _ServerStartupError(RuntimeError):
    """The server subprocess failed to come up for a reason unrelated to a
    port race (import error, index load failure, etc.)."""


def _free_port() -> int:
    """Ask the OS for a currently-free ephemeral port by binding to port 0.

    There is an inherent TOCTOU race (the port could be grabbed between this
    call returning and the subprocess binding it) -- handled by retrying in
    `rag_server` on a detected port conflict, per the task's hard constraint.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(proc: subprocess.Popen, base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{base_url}/health"
    while time.monotonic() < deadline:
        ret = proc.poll()
        if ret is not None:
            output = proc.stdout.read() if proc.stdout else ""
            if "address already in use" in output.lower():
                raise _PortConflict(output[-2000:])
            raise _ServerStartupError(
                f"server process exited (code {ret}) before answering "
                f"{health_url}:\n{output[-4000:]}"
            )
        try:
            r = httpx.get(health_url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise _ServerStartupError(
        f"server did not answer {health_url} within {timeout}s (index loading "
        "may be slower than expected -- see server stdout in the failure above)"
    )


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    """Terminate ONLY the exact subprocess we started -- never any other PID."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture(scope="session")
def rag_server():
    """Boot a private PaperFetchServer on 127.0.0.1:<ephemeral>, yield its
    base URL, and always tear it down. Never an already-running server."""
    proc: Optional[subprocess.Popen] = None
    base_url: Optional[str] = None

    for _attempt in range(5):
        port = _free_port()
        candidate_url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paper_fetch_mcp",
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_health(proc, candidate_url, SERVER_BOOT_TIMEOUT_S)
            base_url = candidate_url
            break
        except _PortConflict:
            _terminate(proc)
            proc = None
            continue
        except _ServerStartupError as e:
            _terminate(proc)
            pytest.fail(f"private smoke-test server failed to start: {e}")
    else:
        pytest.fail("could not find a free ephemeral port for the smoke-test server after 5 attempts")

    try:
        yield base_url
    finally:
        _terminate(proc)


# ---------------------------------------------------------------------------
# MCP call helpers
# ---------------------------------------------------------------------------


async def _call_tool_async(base_url: str, name: str, args: Dict[str, Any]) -> str:
    async with Client(f"{base_url}/mcp") as client:
        result = await client.call_tool(name, args)
        # Tools return a single YAML string through output.respond(); fastmcp
        # surfaces that as `.data` for a str-returning tool.
        assert isinstance(result.data, str), (
            f"{name} returned {type(result.data)!r}, expected the YAML string "
            "output.respond() produces"
        )
        return result.data


def _call_tool(base_url: str, name: str, args: Dict[str, Any]) -> str:
    return asyncio.run(_call_tool_async(base_url, name, args))


def _parse(text: str) -> Dict[str, Any]:
    data = yaml.safe_load(text)
    assert isinstance(data, dict), f"expected a YAML mapping envelope, got {type(data)!r}: {text[:200]!r}"
    return data


def _assert_under_hard_cap(label: str, text: str) -> None:
    assert len(text) <= HARD_CAP, (
        f"{label} response is {len(text)} chars, exceeds the {HARD_CAP}-char "
        "hard cap"
    )


# ---------------------------------------------------------------------------
# Smoke assertions
# ---------------------------------------------------------------------------

REMOVED_SEARCH_TOOLS = {
    "hybrid_search",
    "hyde_query",
    "stepback_query",
    "list_available_tags",
    "reload_indices",
}


def test_server_status_is_alive(rag_server):
    """Basic liveness: the server answers a cheap tool call at all."""
    text = _call_tool(rag_server, "server_status", {})
    _assert_under_hard_cap("server_status", text)
    data = _parse(text)
    assert data["server"] == "PaperFetchServer"
    assert "database" in data and "jobs" in data


def test_live_tool_list_has_no_corpus_search_tools(rag_server):
    async def _names():
        async with Client(f"{rag_server}/mcp") as client:
            return {t.name for t in await client.list_tools()}

    names = asyncio.run(_names())
    assert "get_paper" in names and "search_metadata" in names
    assert not (names & REMOVED_SEARCH_TOOLS)


def test_health_endpoint_reports_db_jobs_telemetry(rag_server):
    checks = httpx.get(f"{rag_server}/health", timeout=10).json()["checks"]
    assert set(checks) == {"db", "jobs", "telemetry"}


def _s2_preflight_ok(timeout: float = 5.0) -> bool:
    """Quick, out-of-band check of whether the real Semantic Scholar API is
    currently reachable and answering (not rate-limited). This host has no
    SEMANTIC_SCHOLAR_API_KEY configured for this test, so the public API's
    per-IP rate limit can be already exhausted by other concurrent users of
    this shared host -- that is an environmental condition, not a defect in
    the server under test, and the task instructions say to skip rather than
    fail the timing assertion in that case."""
    try:
        r = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": "smoke test preflight", "limit": 1},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def test_semantic_scholar_search_answers(rag_server):
    s2_text = _call_tool(rag_server, "semantic_scholar_search", {"query": QUERY, "limit": 3})
    _assert_under_hard_cap("semantic_scholar_search", s2_text)

    s2_data = _parse(s2_text)
    s2_is_upstream_error = s2_data.get("kind") == "upstream_error" or "error" in s2_data

    if s2_is_upstream_error and not _s2_preflight_ok():
        pytest.skip(
            "Semantic Scholar API is not reachable/is rate-limiting this host right "
            f"now (s2 envelope: {s2_data}) -- environmental; skipping"
        )

    assert not s2_is_upstream_error, f"semantic_scholar_search returned an error envelope: {s2_data}"
    assert s2_data.get("papers"), f"semantic_scholar_search returned no papers: {s2_data}"
