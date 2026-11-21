"""FastMCP app assembly for PaperFetchServer.

`build()` is the only thing callers (`__main__.py`, tests) should call:

1. It installs a FastMCP lifespan on the shared `paper_fetch_mcp.mcp`
   singleton that does ALL startup work -- `papers.init_fts()`,
   `publisher.ensure_cache_table()`, `jobs.init_db()`, the job worker,
   `telemetry.init_db()` -- so none of that happens at import time
   (core/__init__.py's "no side effects at import" rule).
2. It adds middleware (telemetry's `CallLogMiddleware`, plus FastMCP's own
   `TimingMiddleware` / `ErrorHandlingMiddleware` / `ResponseLimitingMiddleware`).
3. It imports the five `tools/*` modules that register the tool surface on
   the shared `mcp` instance (deferred imports, so `paper_fetch_mcp.app`
   itself stays side-effect-free at import time; only calling `build()`
   registers tools and installs the lifespan).
4. It registers `/health` with probes over the papers DB, the job queue
   and telemetry.

The papers DB (`config.DB_PATH`) belongs to the corpus, which lives outside
this repository (see `config.CORPUS_ROOT`). When it is absent the server
still starts: the two DB startup steps log a warning, corpus-backed tools
report misses, and the online tools work normally.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from paper_fetch_mcp import mcp
from paper_fetch_mcp.core import jobs as jobs_mod
from paper_fetch_mcp.core import papers as papers_mod
from paper_fetch_mcp.core import publisher as publisher_mod
from paper_fetch_mcp.core import telemetry as telemetry_mod

logger = logging.getLogger(__name__)

# Guards build() so calling it twice (e.g. once from a test, once from
# __main__) doesn't try to re-register the same custom /health route or
# re-add the same middleware instances.
_built = False


# ---------------------------------------------------------------------------
# Health probes -- cheap, best-effort; register_health() already wraps each
# probe in its own try/except so a failing probe degrades "status" rather
# than crashing /health.
# ---------------------------------------------------------------------------


def _health_probe_db() -> Dict[str, Any]:
    with papers_mod.get_db_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM papers")
        return {"reachable": True, "papers": cur.fetchone()[0]}


def _health_probe_jobs() -> Dict[str, Any]:
    jobs = jobs_mod.list_jobs(limit=20)
    queued = sum(1 for j in jobs if j.get("status") == "queued")
    worker = jobs_mod.get_worker()
    return {
        "recent": len(jobs),
        "queued": queued,
        "worker_alive": bool(worker and worker.is_alive()),
    }


def _health_probe_telemetry() -> Dict[str, Any]:
    row = telemetry_mod.last_call_row()
    return {"last_call": row}


# ---------------------------------------------------------------------------
# ClientDisconnect noise. A client that closes an HTTP connection mid-request/response
# is routine SSE/HTTP churn, not a server bug: the MCP SDK's
# `mcp.server.streamable_http` logger currently logs it via
# `logger.exception("Error handling POST request")`, which prints a full
# traceback for every occurrence. This filter mutates ONLY records whose
# exception is (or wraps) `starlette.requests.ClientDisconnect` down to one
# WARN line with no traceback; every other error still logs exactly as
# before (full traceback, original level).
# ---------------------------------------------------------------------------


class _ClientDisconnectQuietFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        exc_info = record.exc_info
        if not exc_info or exc_info[0] is None:
            return True
        try:
            from starlette.requests import ClientDisconnect
        except Exception:
            return True
        exc = exc_info[1]
        if not isinstance(exc, ClientDisconnect):
            return True

        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        record.msg = "client disconnected mid-request (ClientDisconnect; traceback suppressed)"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _quiet_client_disconnect_tracebacks() -> None:
    """Idempotent: attaches the filter at most once per logger."""
    for name in ("mcp.server.streamable_http", "mcp.server.lowlevel.server"):
        target = logging.getLogger(name)
        if not any(isinstance(f, _ClientDisconnectQuietFilter) for f in target.filters):
            target.addFilter(_ClientDisconnectQuietFilter())


# ---------------------------------------------------------------------------
# Lifespan -- ALL loading happens here, never at import time.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(server) -> AsyncIterator[Dict[str, Any]]:
    logger.info("PaperFetchServer: lifespan startup -- initialising DBs...")

    papers_mod.init_fts()
    publisher_mod.ensure_cache_table()
    jobs_mod.init_db()
    # A row can be left status="running" if the previous process died or
    # was killed mid-job -- reconcile those to "error" BEFORE starting the
    # worker, so job_status never polls a dead job as "running" forever.
    stale = jobs_mod.reconcile_stale_jobs()
    if stale:
        logger.warning("jobs: reconciled %d stale 'running' job(s) to 'error'", stale)
    # Process-wide concurrency of 1 for subprocess jobs falls out
    # of there being exactly one JobWorker thread; see core/jobs.py.
    jobs_mod.start_worker()
    telemetry_mod.init_db()

    logger.info("PaperFetchServer: lifespan startup complete")
    try:
        yield {}
    finally:
        logger.info("PaperFetchServer: lifespan shutdown")
        # Join the worker thread before returning so shutdown never leaves
        # it orphaned or hangs indefinitely (stop() has its own timeout).
        jobs_mod.stop_worker()


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


def build():
    """Assemble and return the shared `paper_fetch_mcp.mcp` FastMCP instance:
    install the lifespan, add middleware, register the tool surface, wire
    `/health`. Idempotent -- a second call is a no-op that just returns the
    already-built `mcp` instance (re-adding the same middleware or
    re-registering the same `/health` route would not be safe).
    """
    global _built
    if _built:
        return mcp
    _built = True

    # FastMCP only accepts a `lifespan=` at construction time (no public
    # setter) but `_lifespan_proxy` (fastmcp/server/server.py) reads
    # `fastmcp_server._lifespan` fresh, dynamically, whenever the underlying
    # low-level server's own lifespan context manager actually runs (i.e. at
    # real server startup, well after this assignment) -- so setting the
    # private attribute here on the existing singleton is safe and preserves
    # object identity (every `from paper_fetch_mcp import mcp` elsewhere
    # keeps working against the exact same instance).
    mcp._lifespan = _lifespan

    _quiet_client_disconnect_tracebacks()

    # fastmcp's own built-in middleware.
    from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
    from fastmcp.server.middleware.response_limiting import ResponseLimitingMiddleware
    from fastmcp.server.middleware.timing import TimingMiddleware

    mcp.add_middleware(ErrorHandlingMiddleware())
    mcp.add_middleware(TimingMiddleware())
    mcp.add_middleware(ResponseLimitingMiddleware())
    mcp.add_middleware(telemetry_mod.CallLogMiddleware())

    # Deferred imports: registers the tool surface as a side effect of
    # import (`@mcp.tool` at module scope).
    import paper_fetch_mcp.tools.admin  # noqa: F401
    import paper_fetch_mcp.tools.content  # noqa: F401
    import paper_fetch_mcp.tools.external  # noqa: F401
    import paper_fetch_mcp.tools.ingest  # noqa: F401
    import paper_fetch_mcp.tools.search  # noqa: F401

    telemetry_mod.register_health(
        mcp,
        probes={
            "db": _health_probe_db,
            "jobs": _health_probe_jobs,
            "telemetry": _health_probe_telemetry,
        },
    )

    return mcp
