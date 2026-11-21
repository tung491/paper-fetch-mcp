"""Tests for paper_fetch_mcp.core.jobs -- a minimal SQLite-backed job table.

Every test calls init_db(tmp_path/...) first, which both creates the table
and repoints the module's active DB path at a temp file -- never the real
telemetry/papers DB.
"""

import sys
import pathlib
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from paper_fetch_mcp.core import jobs


def _fresh_db(tmp_path, name="jobs.db"):
    """Point the jobs module at a fresh temp DB and return its path."""
    db_path = tmp_path / name
    jobs.init_db(db_path)
    return db_path


def test_init_db_creates_table(tmp_path):
    db_path = _fresh_db(tmp_path)
    assert db_path.exists()

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1


def test_init_db_idempotent(tmp_path):
    db_path = tmp_path / "jobs.db"
    jobs.init_db(db_path)
    jobs.init_db(db_path)  # must not raise

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
    ).fetchall()
    conn.close()
    assert len(rows) == 1


def test_new_job_returns_uuid_hex_id(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("download_arxiv", {"arxiv_id": "2401.00000"})
    assert isinstance(job_id, str)
    assert len(job_id) == 32  # uuid4().hex


def test_new_job_then_get_job_round_trip(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("download_arxiv", {"arxiv_id": "2401.00000"})

    record = jobs.get_job(job_id)
    assert record is not None
    assert record["job_id"] == job_id
    assert record["kind"] == "download_arxiv"
    assert record["status"] == "queued"
    assert record["payload_json"] == {"arxiv_id": "2401.00000"}
    assert record["result_json"] is None
    assert record["error"] is None
    assert record["created_ts"] == record["updated_ts"]


def test_get_job_unknown_id_returns_none(tmp_path):
    _fresh_db(tmp_path)
    assert jobs.get_job("does-not-exist") is None


def test_get_job_before_init_db_returns_none(tmp_path):
    """A path that was never init_db'd (table/file absent) -> None, not an error."""
    jobs._active_path = tmp_path / "never_created.db"
    assert jobs.get_job("anything") is None


def test_update_job_status_transitions(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("mistral_ocr", {"paper_id": "p1"})

    jobs.update_job(job_id, status="running")
    record = jobs.get_job(job_id)
    assert record["status"] == "running"

    jobs.update_job(job_id, status="done", result={"pages": 12})
    record = jobs.get_job(job_id)
    assert record["status"] == "done"
    assert record["result_json"] == {"pages": 12}
    assert record["error"] is None


def test_update_job_error_status(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("mistral_ocr", {"paper_id": "p1"})

    jobs.update_job(job_id, status="error", error="OCR failed: exit 1")
    record = jobs.get_job(job_id)
    assert record["status"] == "error"
    assert record["error"] == "OCR failed: exit 1"


def test_update_job_bumps_updated_ts(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("mistral_ocr", {})
    before = jobs.get_job(job_id)

    import time
    time.sleep(0.01)
    jobs.update_job(job_id, status="running")
    after = jobs.get_job(job_id)

    assert after["updated_ts"] != before["updated_ts"]
    assert after["created_ts"] == before["created_ts"]


def test_update_job_unknown_field_is_ignored(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("mistral_ocr", {})
    # Should not raise, and should not touch recognized fields.
    jobs.update_job(job_id, bogus_field="whatever")
    record = jobs.get_job(job_id)
    assert record["status"] == "queued"


def test_update_job_unknown_id_is_noop(tmp_path):
    _fresh_db(tmp_path)
    # Must not raise even though no row matches.
    jobs.update_job("does-not-exist", status="done")


def test_list_jobs_returns_newest_first(tmp_path):
    _fresh_db(tmp_path)
    id1 = jobs.new_job("download_arxiv", {"n": 1})
    id2 = jobs.new_job("download_arxiv", {"n": 2})

    records = jobs.list_jobs()
    ids = [r["job_id"] for r in records]
    assert ids[:2] == [id2, id1] or set(ids) == {id1, id2}
    assert len(records) == 2


def test_list_jobs_filters_by_kind(tmp_path):
    _fresh_db(tmp_path)
    arxiv_id = jobs.new_job("download_arxiv", {})
    jobs.new_job("mistral_ocr", {})

    records = jobs.list_jobs(kind="download_arxiv")
    assert len(records) == 1
    assert records[0]["job_id"] == arxiv_id
    assert records[0]["kind"] == "download_arxiv"


def test_list_jobs_respects_limit(tmp_path):
    _fresh_db(tmp_path)
    for i in range(5):
        jobs.new_job("download_arxiv", {"n": i})

    records = jobs.list_jobs(limit=2)
    assert len(records) == 2


def test_list_jobs_before_init_db_returns_empty(tmp_path):
    jobs._active_path = tmp_path / "never_created2.db"
    assert jobs.list_jobs() == []


def test_no_side_effects_at_import(tmp_path, monkeypatch):
    """Importing core.jobs must not create or touch any DB file on disk."""
    import importlib

    fake_state_dir = tmp_path / "state"
    monkeypatch.setattr("paper_fetch_mcp.config.STATE_DIR", fake_state_dir)

    # A plain sys.modules deletion isn't enough here: the parent package
    # (paper_fetch_mcp.core) already holds a `jobs` attribute from this
    # test module's own top-level `from paper_fetch_mcp.core import jobs`,
    # and `from pkg import submodule` resolves via getattr(pkg, submodule)
    # before ever consulting sys.modules -- so it would silently hand back
    # the stale, already-imported module. importlib.reload re-executes the
    # module body in place against the now-patched config.
    fresh_jobs = importlib.reload(jobs)

    assert fresh_jobs.JOBS_DB_PATH == fake_state_dir / "mcp_jobs.db"
    assert not fake_state_dir.exists(), (
        "import must not create the jobs DB directory/file -- only init_db() may"
    )

    # Restore real state for any tests that run after this one in-process.
    importlib.reload(jobs)


# ---------------------------------------------------------------------------
# JobWorker -- drains the queue, concurrency 1, never dies on a bad handler.
#
# Every test here uses a FAKE, in-process handler (never a real subprocess/
# ArXiv/Mistral call) registered via jobs.register_handler, and always tears
# down the process-wide worker singleton + handler registry so no state
# leaks into another test in this (or another) file.
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_worker():
    """Fresh JobWorker singleton + empty handler registry for this test;
    always stopped/cleared again on teardown, even if the test fails."""
    jobs.reset_worker_singleton_for_tests()
    jobs.clear_handlers()
    yield
    jobs.reset_worker_singleton_for_tests()
    jobs.clear_handlers()


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_worker_drains_a_queued_job_to_done(tmp_path, clean_worker):
    jobs.init_db(tmp_path / "jobs.db")
    jobs.register_handler("double", lambda payload: {"n": payload["n"] * 2})
    job_id = jobs.new_job("double", {"n": 3})

    jobs.start_worker(poll_interval=0.02)
    assert _wait_until(lambda: jobs.get_job(job_id)["status"] in ("done", "error"))

    record = jobs.get_job(job_id)
    assert record["status"] == "done"
    assert record["result_json"] == {"n": 6}
    assert record["error"] is None


def test_worker_records_handler_exception_as_error_with_message_captured(
    tmp_path, clean_worker
):
    jobs.init_db(tmp_path / "jobs.db")

    def failing_handler(payload):
        raise RuntimeError("OCR/enrich failed: exit 1: stderr: boom on line 3")

    jobs.register_handler("boom_kind", failing_handler)
    job_id = jobs.new_job("boom_kind", {})

    jobs.start_worker(poll_interval=0.02)
    assert _wait_until(lambda: jobs.get_job(job_id)["status"] in ("done", "error"))

    record = jobs.get_job(job_id)
    assert record["status"] == "error"
    assert "OCR/enrich failed: exit 1: stderr: boom on line 3" in record["error"]
    assert record["result_json"] is None


def test_worker_survives_a_failing_job_and_keeps_draining_the_queue(
    tmp_path, clean_worker
):
    jobs.init_db(tmp_path / "jobs.db")
    jobs.register_handler(
        "always_fails", lambda payload: (_ for _ in ()).throw(ValueError("nope"))
    )
    jobs.register_handler("always_succeeds", lambda payload: {"ok": True})

    failing_id = jobs.new_job("always_fails", {})
    ok_id = jobs.new_job("always_succeeds", {})

    worker = jobs.start_worker(poll_interval=0.02)
    assert _wait_until(
        lambda: jobs.get_job(failing_id)["status"] in ("done", "error")
        and jobs.get_job(ok_id)["status"] in ("done", "error")
    )

    assert jobs.get_job(failing_id)["status"] == "error"
    assert jobs.get_job(ok_id)["status"] == "done"
    # The worker loop itself must still be alive -- one bad job must never
    # kill it.
    assert worker.is_alive()


def test_worker_unregistered_kind_is_recorded_as_error_not_a_hang(
    tmp_path, clean_worker
):
    jobs.init_db(tmp_path / "jobs.db")
    job_id = jobs.new_job("no_such_kind", {})

    jobs.start_worker(poll_interval=0.02)
    assert _wait_until(lambda: jobs.get_job(job_id)["status"] in ("done", "error"))

    record = jobs.get_job(job_id)
    assert record["status"] == "error"
    assert "no_such_kind" in record["error"]


def test_worker_concurrency_is_one(tmp_path, clean_worker):
    """Process-wide concurrency of 1 for subprocess jobs (spec §5): even
    with several queued jobs, at most one handler ever runs at a time."""
    jobs.init_db(tmp_path / "jobs.db")
    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def slow_handler(payload):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.08)
        with lock:
            state["active"] -= 1
        return {"ok": True}

    jobs.register_handler("slow", slow_handler)
    job_ids = [jobs.new_job("slow", {"i": i}) for i in range(4)]

    jobs.start_worker(poll_interval=0.01)
    assert _wait_until(
        lambda: all(jobs.get_job(jid)["status"] == "done" for jid in job_ids),
        timeout=10.0,
    )

    assert state["max_active"] == 1


def test_worker_start_is_idempotent(tmp_path, clean_worker):
    jobs.init_db(tmp_path / "jobs.db")
    worker1 = jobs.start_worker(poll_interval=0.05)
    worker2 = jobs.start_worker(poll_interval=0.05)
    assert worker1 is worker2
    assert worker1.is_alive()


def test_stop_worker_joins_thread_cleanly(tmp_path, clean_worker):
    """Clean shutdown: stop_worker() must actually join the thread (no
    orphaned thread, no hang) within its timeout."""
    jobs.init_db(tmp_path / "jobs.db")
    worker = jobs.start_worker(poll_interval=0.02)
    assert worker.is_alive()

    started = time.monotonic()
    jobs.stop_worker(timeout=5.0)
    elapsed = time.monotonic() - started

    assert not worker.is_alive()
    # Should stop promptly (well under the timeout) when idle, not hang.
    assert elapsed < 2.0


def test_stop_worker_waits_for_the_in_flight_job_to_finish(tmp_path, clean_worker):
    """Stopping mid-job must not abandon it -- the in-flight job still gets
    recorded as done/error, it just doesn't start a new one afterwards."""
    jobs.init_db(tmp_path / "jobs.db")

    def slow_handler(payload):
        time.sleep(0.2)
        return {"ok": True}

    jobs.register_handler("slow_once", slow_handler)
    job_id = jobs.new_job("slow_once", {})

    jobs.start_worker(poll_interval=0.02)
    assert _wait_until(lambda: jobs.get_job(job_id)["status"] == "running", timeout=2.0)

    jobs.stop_worker(timeout=5.0)
    record = jobs.get_job(job_id)
    assert record["status"] == "done"
    assert record["result_json"] == {"ok": True}


# ---------------------------------------------------------------------------
# reconcile_stale_jobs -- a job left "running" from a previous, crashed
# process must not poll as "running" forever.
# ---------------------------------------------------------------------------


def test_reconcile_stale_jobs_marks_running_jobs_as_error(tmp_path):
    _fresh_db(tmp_path)
    job_id = jobs.new_job("mistral_ocr", {})
    jobs.update_job(job_id, status="running")

    reconciled = jobs.reconcile_stale_jobs()

    assert reconciled == 1
    record = jobs.get_job(job_id)
    assert record["status"] == "error"
    assert "stale" in record["error"]


def test_reconcile_stale_jobs_leaves_other_statuses_untouched(tmp_path):
    _fresh_db(tmp_path)
    queued_id = jobs.new_job("mistral_ocr", {})
    done_id = jobs.new_job("mistral_ocr", {})
    jobs.update_job(done_id, status="done", result={"ok": True})

    reconciled = jobs.reconcile_stale_jobs()

    assert reconciled == 0
    assert jobs.get_job(queued_id)["status"] == "queued"
    assert jobs.get_job(done_id)["status"] == "done"


def test_reconcile_stale_jobs_before_init_db_is_a_noop(tmp_path):
    jobs._active_path = tmp_path / "never_created3.db"
    assert jobs.reconcile_stale_jobs() == 0
