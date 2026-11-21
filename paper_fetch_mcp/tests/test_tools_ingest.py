"""Tests for the ingest_arxiv / job_status / download_arxiv @mcp.tool
wrappers in tools/ingest.py (design spec
docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md, §6,
§8 Phase 2).

All three tools must stay THIN wrappers over `core.ingest.enqueue_ingest_job`
/ `core.jobs.get_job` -- none of them may run arxiv/subprocess/OCR logic
itself. Every case here mocks those core functions, so this file never
touches the real corpus, the real job DB, or the network.
"""

from __future__ import annotations

import sys
import pathlib
from unittest.mock import patch

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from paper_fetch_mcp.tools import ingest as ingest_tool


# ---------------------------------------------------------------------------
# ingest_arxiv
# ---------------------------------------------------------------------------


def test_ingest_arxiv_returns_job_id_and_queued_status_immediately():
    with patch.object(
        ingest_tool.ingest_mod,
        "enqueue_ingest_job",
        return_value={"job_id": "abc123", "status": "queued"},
    ) as mock_enqueue:
        out = ingest_tool.ingest_arxiv(ids=["1234.56789"])
        mock_enqueue.assert_called_once_with(["1234.56789"])

    data = yaml.safe_load(out)
    assert data["job_id"] == "abc123"
    assert data["status"] == "queued"


def test_ingest_arxiv_tool_is_a_plain_def_not_async():
    import inspect

    assert not inspect.iscoroutinefunction(ingest_tool.ingest_arxiv)


# ---------------------------------------------------------------------------
# job_status
# ---------------------------------------------------------------------------


def test_job_status_returns_the_job_record():
    fake_record = {
        "job_id": "abc123",
        "kind": "ingest_arxiv",
        "status": "done",
        "payload_json": {"ids": ["1234.56789"]},
        "result_json": {"Doe2024A": "content"},
        "error": None,
        "created_ts": "2026-09-13T00:00:00+00:00",
        "updated_ts": "2026-09-13T00:00:05+00:00",
    }
    with patch.object(
        ingest_tool.jobs_mod, "get_job", return_value=fake_record
    ) as mock_get_job:
        out = ingest_tool.job_status(job_id="abc123")
        mock_get_job.assert_called_once_with("abc123")

    data = yaml.safe_load(out)
    assert data["status"] == "done"
    assert data["result_json"] == {"Doe2024A": "content"}


def test_job_status_reports_captured_error_verbatim_on_failure():
    fake_record = {
        "job_id": "abc123",
        "kind": "ingest_arxiv",
        "status": "error",
        "payload_json": {"ids": ["9999.99999"]},
        "result_json": None,
        "error": "RuntimeError: Processing failed during OCR/enrich step "
        "(command=['uv', 'run', 'python', 'mistral_ocr.py'], exit=1). "
        "stderr: some real stderr text",
        "created_ts": "2026-09-13T00:00:00+00:00",
        "updated_ts": "2026-09-13T00:00:05+00:00",
    }
    with patch.object(ingest_tool.jobs_mod, "get_job", return_value=fake_record):
        out = ingest_tool.job_status(job_id="abc123")

    data = yaml.safe_load(out)
    assert data["status"] == "error"
    assert "some real stderr text" in data["error"]


def test_job_status_unknown_job_id_reports_found_false():
    with patch.object(ingest_tool.jobs_mod, "get_job", return_value=None):
        out = ingest_tool.job_status(job_id="does-not-exist")

    data = yaml.safe_load(out)
    assert data["found"] is False
    assert data["job_id"] == "does-not-exist"


def test_job_status_tool_is_a_plain_def_not_async():
    import inspect

    assert not inspect.iscoroutinefunction(ingest_tool.job_status)


# ---------------------------------------------------------------------------
# download_arxiv -- thin alias (spec §4 non-goal: old names stay as thin
# aliases). Must keep its original signature (ids: List[str]) but now
# returns the job record instead of blocking for OCR'd content.
# ---------------------------------------------------------------------------


def test_download_arxiv_alias_enqueues_and_returns_the_job_record():
    fake_record = {
        "job_id": "abc123",
        "kind": "ingest_arxiv",
        "status": "queued",
        "payload_json": {"ids": ["1234.56789"]},
        "result_json": None,
        "error": None,
        "created_ts": "2026-09-13T00:00:00+00:00",
        "updated_ts": "2026-09-13T00:00:00+00:00",
    }
    with patch.object(
        ingest_tool.ingest_mod,
        "enqueue_ingest_job",
        return_value={"job_id": "abc123", "status": "queued"},
    ) as mock_enqueue, patch.object(
        ingest_tool.jobs_mod, "get_job", return_value=fake_record
    ) as mock_get_job:
        out = ingest_tool.download_arxiv(ids=["1234.56789"])
        mock_enqueue.assert_called_once_with(["1234.56789"])
        mock_get_job.assert_called_once_with("abc123")

    data = yaml.safe_load(out)
    assert data["job_id"] == "abc123"
    assert data["kind"] == "ingest_arxiv"
    assert data["status"] == "queued"


def test_download_arxiv_alias_falls_back_to_bare_outcome_if_job_vanished():
    """Defensive: if get_job somehow can't find the job it just created
    (race/DB hiccup), don't crash -- fall back to the {job_id, status} the
    enqueue call itself returned."""
    with patch.object(
        ingest_tool.ingest_mod,
        "enqueue_ingest_job",
        return_value={"job_id": "abc123", "status": "queued"},
    ), patch.object(ingest_tool.jobs_mod, "get_job", return_value=None):
        out = ingest_tool.download_arxiv(ids=["1234.56789"])

    data = yaml.safe_load(out)
    assert data["job_id"] == "abc123"
    assert data["status"] == "queued"


def test_download_arxiv_tool_is_a_plain_def_not_async():
    import inspect

    assert not inspect.iscoroutinefunction(ingest_tool.download_arxiv)


def test_download_arxiv_keeps_its_original_ids_signature():
    import inspect

    sig = inspect.signature(ingest_tool.download_arxiv)
    assert list(sig.parameters) == ["ids"]
