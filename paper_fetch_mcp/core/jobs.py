"""SQLite-backed background job table + worker (spec §5 "Target
architecture", §8 Phase 2, finding F9).

Phase 1 left this module intentionally inert (a schema plus five functions,
no worker) because nothing drained the queue yet: a wave briefly made
``download_arxiv`` enqueue into it and return a plausible ``job_id`` that
NOTHING ever processed -- a black hole, worse than the old synchronous
behaviour it replaced (that regression was reverted in ``core/ingest.py``,
see its docstring). Phase 2 (this wave) is the real consumer, so this
module now also owns:

- A handler registry (``register_handler(kind, fn)``): the worker doesn't
  know how to run any particular kind of job itself -- ``tools/ingest.py``
  registers the ``"ingest_arxiv"`` handler at its own import time, the same
  way it registers its ``@mcp.tool`` wrappers. ``core/jobs.py`` stays
  generic on purpose so the ``kg_*`` plan (spec §9) can share this exact
  table/worker for its own job kinds without editing this file.
- ``JobWorker``: one background thread that claims the oldest ``queued``
  job, runs its handler to completion, records ``done``/``error``, then
  looks for the next one. Concurrency is exactly 1 by construction -- there
  is only ever one worker thread, so at most one job (subprocess or not)
  ever runs at a time process-wide (spec §5: "subprocess jobs 1"). A
  handler that raises is caught and recorded as that job's `error` (str(exc)
  verbatim); it never kills the worker loop.
- ``start_worker()`` / ``stop_worker()``: a process-wide singleton, started
  from the FastMCP lifespan in ``app.py`` (never at import) and stopped
  there too, so shutdown joins the thread instead of leaving it orphaned.
- ``reconcile_stale_jobs()``: called once at startup, after ``init_db()``
  and before ``start_worker()``. A job can be left ``status="running"`` if
  the previous process died or was killed mid-job; nothing would ever pick
  such a row up again (the worker only claims ``"queued"`` rows), so left
  alone it would poll as ``"running"`` forever. This marks every such row
  ``"error"`` with a clearly-labelled stale-job message.

Path-resolution style matches ``core/telemetry.py``: a module-level default
path under ``STATE_DIR``. ``init_db(path=...)`` both creates the
table at that path AND becomes the active path for every other function in
this module (so tests can point the whole module at a temp DB by calling
``init_db(tmp_path)`` first, without every function taking a `path` kwarg --
that keeps the public signatures exactly as specified). ``init_db`` is never
called at import time -- the FastMCP lifespan calls it, per
``core/__init__.py``'s "no side effects at import" rule.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from paper_fetch_mcp.config import STATE_DIR

logger = logging.getLogger(__name__)

JOBS_DB_PATH = STATE_DIR / "mcp_jobs.db"

# The active DB path for this module. Defaults to JOBS_DB_PATH; init_db(path)
# repoints it (used by tests to isolate against a temp DB).
_active_path: Path = JOBS_DB_PATH

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    payload_json TEXT,
    result_json  TEXT,
    error        TEXT,
    created_ts   TEXT NOT NULL,
    updated_ts   TEXT NOT NULL
)
"""
_CREATE_INDEX_KIND_SQL = "CREATE INDEX IF NOT EXISTS idx_jobs_kind ON jobs(kind)"

# Guards every write to the jobs DB (coarse but simple; matches telemetry.py).
_write_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_active_path), timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Optional[Union[str, Path]] = None) -> None:
    """Create the jobs table (+ index) if absent, and enable WAL mode.

    If `path` is given, it becomes the active DB path for every other
    function in this module (new_job/get_job/update_job/list_jobs) until
    init_db is called again. Call this from the FastMCP lifespan -- never at
    import time.
    """
    global _active_path
    _active_path = Path(path) if path is not None else JOBS_DB_PATH
    _active_path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        conn = sqlite3.connect(str(_active_path), timeout=5, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_KIND_SQL)
            conn.commit()
        finally:
            conn.close()


def _row_to_job(row: sqlite3.Row) -> Dict[str, Any]:
    d = {k: row[k] for k in row.keys()}
    d["payload_json"] = json.loads(d["payload_json"]) if d.get("payload_json") else None
    d["result_json"] = json.loads(d["result_json"]) if d.get("result_json") else None
    return d


def new_job(kind: str, payload: Dict[str, Any]) -> str:
    """Create a queued job of `kind` with `payload`, and return its job_id."""
    job_id = uuid.uuid4().hex
    ts = _now()
    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO jobs "
                "(job_id, kind, status, payload_json, result_json, error, created_ts, updated_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, kind, "queued", json.dumps(payload), None, None, ts, ts),
            )
            conn.commit()
        finally:
            conn.close()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job record for `job_id`, or None if it does not exist."""
    if not _active_path.exists():
        return None
    with _write_lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        finally:
            conn.close()
    return _row_to_job(row) if row is not None else None


# Job fields accepted by update_job(**fields), mapped to their DB column.
# `payload` / `result` are JSON-encoded automatically -- pass the raw
# dict/list, not an already-serialized JSON string.
_UPDATE_COLUMN_MAP = {
    "status": "status",
    "payload": "payload_json",
    "result": "result_json",
    "error": "error",
}
_JSON_FIELDS = {"payload", "result"}


def update_job(job_id: str, **fields: Any) -> None:
    """Merge `fields` into the job record for `job_id` and bump updated_ts.

    Recognized fields: status, payload, result, error. Unknown field names
    are ignored. No-op (does not touch updated_ts) if no recognized field is
    passed.
    """
    set_clauses = []
    values: List[Any] = []
    for name, value in fields.items():
        column = _UPDATE_COLUMN_MAP.get(name)
        if column is None:
            continue
        if name in _JSON_FIELDS:
            value = json.dumps(value) if value is not None else None
        set_clauses.append(f"{column} = ?")
        values.append(value)

    if not set_clauses:
        return

    set_clauses.append("updated_ts = ?")
    values.append(_now())
    values.append(job_id)

    with _write_lock:
        conn = _connect()
        try:
            conn.execute(
                f"UPDATE jobs SET {', '.join(set_clauses)} WHERE job_id = ?",
                values,
            )
            conn.commit()
        finally:
            conn.close()


def list_jobs(kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent `limit` jobs, newest first, optionally filtered by `kind`."""
    if not _active_path.exists():
        return []
    with _write_lock:
        conn = _connect()
        try:
            if kind is not None:
                cur = conn.execute(
                    "SELECT * FROM jobs WHERE kind = ? ORDER BY created_ts DESC LIMIT ?",
                    (kind, limit),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_ts DESC LIMIT ?", (limit,)
                )
            rows = cur.fetchall()
        finally:
            conn.close()
    return [_row_to_job(r) for r in rows]


# ---------------------------------------------------------------------------
# Stale-job reconciliation (spec §8 Phase 2: "job left running from a
# previous process is reconciled at startup").
# ---------------------------------------------------------------------------

STALE_JOB_ERROR = (
    "stale: job was still 'running' when the server (re)started -- the "
    "process that was executing it is gone, so its outcome is unknown"
)


def reconcile_stale_jobs() -> int:
    """Mark every job left ``status="running"`` as ``status="error"``.

    A row can be stuck ``"running"`` only if the previous process died or
    was killed mid-job (a clean shutdown always drains via `stop_worker()`,
    which lets the in-flight job finish first -- see `JobWorker.stop`).
    Nothing would ever revisit such a row on its own (the worker only
    claims ``"queued"`` rows), so left alone `job_status` would report
    ``"running"`` forever. Call this once at startup, right after
    `init_db()` and BEFORE `start_worker()`.

    Returns the number of rows reconciled. A no-op (returns 0) if the DB
    file doesn't exist yet.
    """
    if not _active_path.exists():
        return 0
    with _write_lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE status = 'running'"
            ).fetchall()
            if not rows:
                return 0
            ts = _now()
            conn.execute(
                "UPDATE jobs SET status = 'error', error = ?, updated_ts = ? "
                "WHERE status = 'running'",
                (STALE_JOB_ERROR, ts),
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Handler registry -- generic on purpose (spec §9: kg_* will share this same
# table/worker for its own job kinds). A handler is `payload_dict ->
# result_dict`; raising signals failure and its message is captured
# verbatim as the job's `error`.
# ---------------------------------------------------------------------------

JobHandler = Callable[[Dict[str, Any]], Dict[str, Any]]

_handlers: Dict[str, JobHandler] = {}


def register_handler(kind: str, handler: JobHandler) -> None:
    """Register the function the worker calls to run a job of `kind`.

    Intended to be called at a `tools/*.py` module's own import time (the
    same pattern as its `@mcp.tool` registrations) -- never at this
    module's import time, which leaves `_handlers` empty.
    """
    _handlers[kind] = handler


def clear_handlers() -> None:
    """Test-only: reset the handler registry between tests."""
    _handlers.clear()


def _claim_next_queued_job() -> Optional[Dict[str, Any]]:
    """Atomically (under `_write_lock`) pick the oldest `status="queued"`
    job, flip it to `"running"`, and return the updated record -- or None
    if the queue is empty."""
    if not _active_path.exists():
        return None
    with _write_lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' "
                "ORDER BY created_ts ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job = _row_to_job(row)
            ts = _now()
            conn.execute(
                "UPDATE jobs SET status = 'running', updated_ts = ? WHERE job_id = ?",
                (ts, job["job_id"]),
            )
            conn.commit()
        finally:
            conn.close()
    job["status"] = "running"
    return job


# ---------------------------------------------------------------------------
# JobWorker -- the background thread. Concurrency is exactly 1 by
# construction (spec §5: "subprocess jobs 1"): there is only ever one
# `JobWorker` thread, and it runs one job to completion before claiming the
# next, so no two jobs (subprocess-backed or not) ever run concurrently in
# this process.
# ---------------------------------------------------------------------------


class JobWorker:
    """Drains the `jobs` queue one job at a time in a background thread."""

    def __init__(self, poll_interval: float = 0.5):
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background thread. Idempotent -- a no-op if already
        running."""
        if self.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="agentic-rag-jobs-worker", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the loop to stop after its current job (if any) finishes,
        and join the thread. Safe to call even if never started, or more
        than once."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "jobs worker did not stop within %.1fs of a shutdown "
                    "signal (likely a job handler still running)",
                    timeout,
                )
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            job = _claim_next_queued_job()
            if job is None:
                # No work: wait_for is interruptible, unlike a plain sleep,
                # so `stop()` doesn't have to wait out a full poll interval.
                self._stop_event.wait(self.poll_interval)
                continue
            self._execute(job)

    def _execute(self, job: Dict[str, Any]) -> None:
        job_id = job["job_id"]
        kind = job["kind"]
        handler = _handlers.get(kind)
        if handler is None:
            update_job(
                job_id,
                status="error",
                error=f"no worker handler registered for kind={kind!r}",
            )
            return
        try:
            result = handler(job.get("payload_json") or {})
        except Exception as exc:  # never let one failing job kill the worker
            error_text = f"{type(exc).__name__}: {exc}"
            logger.error(
                "job %s (kind=%s) failed: %s", job_id, kind, error_text, exc_info=True
            )
            update_job(job_id, status="error", error=error_text)
            return
        update_job(job_id, status="done", result=result)


# ---------------------------------------------------------------------------
# Process-wide singleton worker, started/stopped from the FastMCP lifespan
# (app.py) -- never at import time.
# ---------------------------------------------------------------------------

_worker: Optional[JobWorker] = None
_worker_lock = threading.Lock()


def start_worker(poll_interval: float = 0.5) -> JobWorker:
    """Start the process-wide `JobWorker` singleton if not already running.
    Idempotent. Call from the FastMCP lifespan, after `init_db()` and
    `reconcile_stale_jobs()`."""
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = JobWorker(poll_interval=poll_interval)
        _worker.start()
        return _worker


def stop_worker(timeout: float = 10.0) -> None:
    """Stop the process-wide `JobWorker` singleton, if one was started.
    Joins its thread so shutdown never leaves it orphaned. Safe to call
    even if `start_worker()` was never called."""
    with _worker_lock:
        if _worker is not None:
            _worker.stop(timeout=timeout)


def get_worker() -> Optional[JobWorker]:
    """Return the process-wide `JobWorker` singleton, or None if
    `start_worker()` has never been called in this process."""
    return _worker


def reset_worker_singleton_for_tests() -> None:
    """Test-only: drop the module-level `_worker` reference (after stopping
    it) so a later `start_worker()` builds a brand new `JobWorker` instead
    of restarting a previous test's stopped one."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop(timeout=10.0)
        _worker = None
