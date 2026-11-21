"""Tests for paper_fetch_mcp.core.ingest.download_arxiv_sync -- the real,
synchronous ArXiv download -> OCR -> enrich -> DB/index pipeline (spec
docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md §8
Phase 1 items 3-5; ported from mcp_server_v2.py:683-807).

These cases were carried over from test_server_v2_hotfix.py's "0.8:
download_arxiv writes BibTeX only after OCR/enrich succeeds" section, which
tested the same behaviour against the monolith directly -- that file now
targets the new stack (see test_server_v2_hotfix.py's module docstring),
and this is the file that actually owns pipeline-body coverage now that the
pipeline lives in core/ingest.py.

No test here ever touches the real corpus (pre_processing_v2/papers.db,
the FAISS indices, or a real subprocess): PROJECT_ROOT is monkeypatched to
a tmp_path, `arxiv.Search` is faked, and `subprocess.run`/`subprocess.Popen`
are faked or monkeypatched to no-ops.
"""

from __future__ import annotations

import sys
import pathlib
from pathlib import Path

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from paper_fetch_mcp.core import ingest
from paper_fetch_mcp.core import jobs as jobs_mod


class _FakeAuthor:
    def __init__(self, name):
        self.name = name


class _FakePublished:
    year = 2024


class _FakePaper:
    def __init__(self, short_id):
        self.authors = [_FakeAuthor("Jane Q. Doe")]
        self.published = _FakePublished()
        self.title = "A Great Paper About Things"
        self.entry_id = f"http://arxiv.org/abs/{short_id}"
        self._short_id = short_id

    def get_short_id(self):
        return self._short_id

    def download_pdf(self, dirpath, filename):
        Path(dirpath, filename).write_bytes(b"%PDF-1.4 fake pdf")


class _FakeSearch:
    def __init__(self, id_list):
        self.id_list = id_list

    def results(self):
        return [_FakePaper(sid) for sid in self.id_list]


def _prep_env(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pre_processing_v2").mkdir(parents=True)
    monkeypatch.setattr(ingest, "arxiv", type("FakeArxivModule", (), {"Search": _FakeSearch}))


def test_download_arxiv_sync_does_not_write_bib_when_ocr_fails(monkeypatch, tmp_path):
    _prep_env(monkeypatch, tmp_path)

    def fake_run(cmd, **kwargs):
        raise ingest.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)

    outcome = ingest.download_arxiv_sync(["1234.56789"])
    assert outcome["ok"] is False
    assert outcome["error"]
    assert not (tmp_path / "references.bib").exists()


def test_download_arxiv_sync_writes_bib_only_after_ocr_succeeds(monkeypatch, tmp_path):
    _prep_env(monkeypatch, tmp_path)

    monkeypatch.setattr(ingest.subprocess, "run", lambda cmd, **kwargs: None)  # OCR + enrich "succeed"
    monkeypatch.setattr(ingest.subprocess, "Popen", lambda *a, **k: None)  # no real background trigger

    outcome = ingest.download_arxiv_sync(["1234.56789"])
    assert outcome["ok"] is True
    bib_path = tmp_path / "references.bib"
    assert bib_path.exists()
    assert "@article" in bib_path.read_text()


def test_download_arxiv_sync_reads_ocr_output_into_results(monkeypatch, tmp_path):
    _prep_env(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest.subprocess, "run", lambda cmd, **kwargs: None)
    monkeypatch.setattr(ingest.subprocess, "Popen", lambda *a, **k: None)

    # Pre-seed the OCR output the fake "OCR run" would have produced --
    # the citation key must match _FakePaper's deterministic construction
    # (firstAuthor + year + firstWordTitle -> "Doe2024A").
    out_dir = tmp_path / "pre_processing_v2" / "mistral_output" / "Doe2024A"
    out_dir.mkdir(parents=True)
    (out_dir / "Doe2024A.md").write_text("# OCR'd content\n\nBody text.")

    outcome = ingest.download_arxiv_sync(["1234.56789"])
    assert outcome["ok"] is True
    assert outcome["results"]["Doe2024A"] == "# OCR'd content\n\nBody text."


def test_download_arxiv_sync_missing_ocr_output_reports_failure_per_key(monkeypatch, tmp_path):
    _prep_env(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest.subprocess, "run", lambda cmd, **kwargs: None)
    monkeypatch.setattr(ingest.subprocess, "Popen", lambda *a, **k: None)

    outcome = ingest.download_arxiv_sync(["1234.56789"])
    assert outcome["ok"] is True
    assert outcome["results"]["Doe2024A"] == "Processing failed: Output file not found."


def test_download_arxiv_sync_caps_ids_at_ten(monkeypatch, tmp_path):
    _prep_env(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest.subprocess, "run", lambda cmd, **kwargs: None)
    monkeypatch.setattr(ingest.subprocess, "Popen", lambda *a, **k: None)

    captured = {}
    real_search_cls = ingest.arxiv.Search

    class CapturingSearch(real_search_cls):
        def __init__(self, id_list):
            captured["id_list"] = list(id_list)
            super().__init__(id_list)

    monkeypatch.setattr(ingest.arxiv, "Search", CapturingSearch)

    ids = [f"1234.{i:05d}" for i in range(15)]
    ingest.download_arxiv_sync(ids)
    assert len(captured["id_list"]) == 10


def test_download_arxiv_sync_arxiv_search_failure_is_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pre_processing_v2").mkdir(parents=True)

    class ExplodingSearch:
        def __init__(self, id_list):
            raise RuntimeError("network unreachable")

    monkeypatch.setattr(ingest, "arxiv", type("FakeArxivModule", (), {"Search": ExplodingSearch}))

    outcome = ingest.download_arxiv_sync(["1234.56789"])
    assert outcome["ok"] is False
    assert "network unreachable" in outcome["error"]


def test_download_arxiv_sync_surfaces_ocr_stderr_verbatim_in_the_error(
    monkeypatch, tmp_path
):
    """spec §11 / finding F9: the pipeline must not swallow the failing
    script's own stderr behind a generic message -- surface it (bounded) so
    the next person can diagnose enrich_metadata.py's exit-1 bug without
    re-running anything. This module still does not attempt to FIX that
    script."""
    _prep_env(monkeypatch, tmp_path)

    def fake_run(cmd, **kwargs):
        raise ingest.subprocess.CalledProcessError(
            1, cmd, output=b"", stderr=b"Traceback: enrich_metadata.py blew up\n"
        )

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)

    outcome = ingest.download_arxiv_sync(["1234.56789"])
    assert outcome["ok"] is False
    assert "enrich_metadata.py blew up" in outcome["error"]
    assert "exit=1" in outcome["error"]


# ---------------------------------------------------------------------------
# Job-model wrapper (spec §6, §8 Phase 2): enqueue_ingest_job / ingest_job_handler
# ---------------------------------------------------------------------------


def test_enqueue_ingest_job_creates_a_queued_job_and_returns_job_id(tmp_path):
    jobs_mod.init_db(tmp_path / "jobs.db")

    outcome = ingest.enqueue_ingest_job(["1234.56789"])

    assert outcome["status"] == "queued"
    assert isinstance(outcome["job_id"], str)
    record = jobs_mod.get_job(outcome["job_id"])
    assert record is not None
    assert record["kind"] == ingest.INGEST_JOB_KIND
    assert record["status"] == "queued"
    assert record["payload_json"] == {"ids": ["1234.56789"]}


def test_enqueue_ingest_job_caps_ids_at_ten(tmp_path):
    jobs_mod.init_db(tmp_path / "jobs.db")

    ids = [f"1234.{i:05d}" for i in range(15)]
    outcome = ingest.enqueue_ingest_job(ids)

    record = jobs_mod.get_job(outcome["job_id"])
    assert len(record["payload_json"]["ids"]) == 10


def test_ingest_job_handler_returns_results_on_success(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "download_arxiv_sync",
        lambda ids: {"ok": True, "results": {"Doe2024A": "content"}},
    )
    result = ingest.ingest_job_handler({"ids": ["1234.56789"]})
    assert result == {"Doe2024A": "content"}


def test_ingest_job_handler_raises_with_the_pipeline_error_message_on_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        ingest,
        "download_arxiv_sync",
        lambda ids: {"ok": False, "error": "Processing failed during OCR/enrich step."},
    )
    import pytest

    with pytest.raises(RuntimeError, match="Processing failed during OCR/enrich step"):
        ingest.ingest_job_handler({"ids": ["1234.56789"]})
