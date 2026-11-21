"""Server-side call history + health endpoint (spec §7 "Telemetry", finding F10).

``CallLogMiddleware`` is a FastMCP middleware that records one row per tool
call into a small SQLite table (``tool_calls``). ``register_health`` adds a
``GET /health`` route reporting process + index/DB/job state.

Telemetry must never break a tool call: every DB write is wrapped in
try/except and logged at WARNING on failure.

Moved here from ``paper_fetch_mcp/telemetry.py`` per the refactor design spec
§5 "Target architecture" (docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md).
"""

from __future__ import annotations

import json
import logging
import os
import re
import resource
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastmcp.server.middleware.middleware import CallNext, Middleware, MiddlewareContext

from paper_fetch_mcp.config import STATE_DIR

logger = logging.getLogger(__name__)

TELEMETRY_DB_PATH = STATE_DIR / "mcp_telemetry.db"

_MAX_ARGS_JSON_CHARS = 2000

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY,
    ts TEXT,
    tool TEXT,
    args_json TEXT,
    duration_ms INTEGER,
    result_chars INTEGER,
    truncated INTEGER,
    error TEXT,
    client_id TEXT
)
"""
_CREATE_INDEX_TS_SQL = "CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(ts)"
_CREATE_INDEX_TOOL_SQL = "CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool)"

# Guards every write to any tool_calls DB (coarse but simple; writes are tiny
# and infrequent relative to a 45s tool timeout budget).
_write_lock = threading.Lock()


def _resolve_path(path: Optional[Union[str, Path]]) -> Path:
    return Path(path) if path is not None else TELEMETRY_DB_PATH


def init_db(path: Optional[Union[str, Path]] = None) -> None:
    """Create the tool_calls table (+ indices) if absent, and enable WAL mode."""
    db_path = _resolve_path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        conn = sqlite3.connect(str(db_path), timeout=5, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_TS_SQL)
            conn.execute(_CREATE_INDEX_TOOL_SQL)
            conn.commit()
        finally:
            conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def recent_calls(limit: int = 50, db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Return the most recent `limit` tool_calls rows, newest first."""
    path = _resolve_path(db_path)
    if not path.exists():
        return []
    with _write_lock:
        conn = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM tool_calls ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = [_row_to_dict(r) for r in cur.fetchall()]
        except sqlite3.Error:
            rows = []
        finally:
            conn.close()
    return rows


def last_call_row(db_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Return the single most recent tool_calls row, or None."""
    rows = recent_calls(limit=1, db_path=db_path)
    return rows[0] if rows else None


def _extract_tool_name(context: MiddlewareContext) -> str:
    try:
        name = getattr(context.message, "name", None)
        if name:
            return str(name)
    except Exception:
        pass
    return str(getattr(context, "method", None) or "unknown")


def _extract_args_json(context: MiddlewareContext) -> str:
    try:
        args = getattr(context.message, "arguments", None) or {}
        text = json.dumps(args, default=str)
    except Exception:
        text = "{}"
    if len(text) > _MAX_ARGS_JSON_CHARS:
        text = text[:_MAX_ARGS_JSON_CHARS]
    return text


def _extract_client_id(context: MiddlewareContext) -> Optional[str]:
    try:
        fctx = context.fastmcp_context
        if fctx is not None:
            return fctx.client_id
    except Exception:
        pass
    return None


def _measure_result(result: Any) -> Tuple[int, bool]:
    """Total chars across a ToolResult's text content blocks, and whether the
    envelope reports itself truncated (best-effort text sniff for
    'truncated: true', as written by output.respond())."""
    parts: List[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    combined = "\n".join(parts)
    truncated = bool(re.search(r"truncated:\s*true", combined, re.IGNORECASE))
    return len(combined), truncated


class CallLogMiddleware(Middleware):
    """FastMCP middleware that logs every tool call into the tool_calls table."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self._db_path = _resolve_path(db_path)

    async def on_call_tool(self, context: MiddlewareContext, call_next: CallNext):
        start = time.perf_counter()
        tool_name = _extract_tool_name(context)
        args_json = _extract_args_json(context)
        client_id = _extract_client_id(context)

        error_text: Optional[str] = None
        result: Any = None
        try:
            result = await call_next(context)
            return result
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration_ms = int((time.perf_counter() - start) * 1000)
            result_chars, truncated = 0, False
            if error_text is None and result is not None:
                try:
                    result_chars, truncated = _measure_result(result)
                except Exception:
                    logger.warning("telemetry: failed to measure result for %s", tool_name, exc_info=True)
            self._record(tool_name, args_json, duration_ms, result_chars, truncated, error_text, client_id)

    def _record(
        self,
        tool: str,
        args_json: str,
        duration_ms: int,
        result_chars: int,
        truncated: bool,
        error_text: Optional[str],
        client_id: Optional[str],
    ) -> None:
        try:
            ts = datetime.now(timezone.utc).isoformat()
            with _write_lock:
                conn = sqlite3.connect(str(self._db_path), timeout=5, check_same_thread=False)
                try:
                    conn.execute(
                        "INSERT INTO tool_calls "
                        "(ts, tool, args_json, duration_ms, result_chars, truncated, error, client_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            ts,
                            tool,
                            args_json,
                            duration_ms,
                            result_chars,
                            int(bool(truncated)),
                            error_text,
                            client_id,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            logger.warning("telemetry: failed to record call for tool=%s", tool, exc_info=True)


def _rss_mb() -> float:
    """Current process RSS in MB. Dependency-light: /proc/self/statm, falling
    back to resource.getrusage's peak RSS."""
    try:
        with open("/proc/self/statm") as f:
            fields = f.read().split()
        resident_pages = int(fields[1])
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, AttributeError):
            page_size = resource.getpagesize()
        return round(resident_pages * page_size / (1024 * 1024), 1)
    except Exception:
        try:
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return round(kb / 1024, 1)
        except Exception:
            return -1.0


def register_health(mcp, *, probes: Optional[Dict[str, Callable[[], Any]]] = None) -> None:
    """Add a GET /health custom route reporting process + probe state.

    `probes` is name -> zero-arg callable returning a JSON-serialisable value.
    Each probe is called in try/except; a failing probe marks the overall
    status "degraded" without raising.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    start_time = time.time()
    probes = probes or {}

    @mcp.custom_route("/health", methods=["GET"])
    async def _health(request: Request) -> JSONResponse:
        status = "ok"
        checks: Dict[str, Any] = {}
        for name, probe in probes.items():
            try:
                checks[name] = probe()
            except Exception as exc:
                checks[name] = {"error": f"{type(exc).__name__}: {exc}"}
                status = "degraded"

        payload = {
            "status": status,
            "pid": os.getpid(),
            "uptime_s": round(time.time() - start_time, 1),
            "rss_mb": _rss_mb(),
            "checks": checks,
        }
        return JSONResponse(payload)
