"""ArXiv ingest pipeline: download -> Mistral OCR -> enrich -> DB/index update.

Design spec `docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md`,
§6, §8 Phase 2, §9, finding F9. Ported, function-for-function, from
`mcp_server_v2.py:683-807` (the monolith's `download_arxiv` body) -- this
module owns the real pipeline body (`download_arxiv_sync`, still fully
synchronous and unchanged in shape) plus, as of Phase 2, the job-model
wrapper around it that lets `tools/ingest.py` expose it asynchronously:

- `download_arxiv_sync(ids)` -- the actual download -> OCR -> enrich ->
  DB/index pipeline. Never raises; always returns `{"ok": bool, ...}`.
- `enqueue_ingest_job(ids)` -- creates a `core.jobs` row of kind
  `INGEST_JOB_KIND` and returns `{"job_id", "status": "queued"}`
  immediately. This is what `ingest_arxiv` and the `download_arxiv` alias
  both call (spec §6); neither tool ever runs the pipeline inline any more.
- `ingest_job_handler(payload)` -- the function `core.jobs`'s worker thread
  calls for every `INGEST_JOB_KIND` job. It runs `download_arxiv_sync`
  and, on `{"ok": False, ...}`, RAISES so the worker records that error
  message verbatim as the job's `error` column (spec §11 / finding F9:
  "ingest_arxiv reports the job error verbatim instead of blocking; fix
  [of enrich_metadata.py] tracked separately" -- this module still makes
  no attempt to fix `enrich_metadata.py`'s own exit-1 bug, only to surface
  it clearly).

Phase 2 regression-fix history this supersedes: Wave 5a's first cut of
`tools/ingest.py:download_arxiv` enqueued a `core.jobs` row and returned
`{job_id, status: "queued"}` immediately, but nothing drained that queue --
there was no worker and no `job_status` tool -- so a caller got a plausible
"success" and nothing ever happened, which was worse than synchronous
behaviour (which at least reported its own failure) -- that regression was
reverted to a fully synchronous `download_arxiv_sync` call for Phase 1.
This wave adds the real worker (`core.jobs.JobWorker`, started from the
FastMCP lifespan in `app.py`) that finally drains what this module enqueues,
so the job model is honest end-to-end.

Ordering preserved from the Phase 0 item 0.8 fix: BibTeX entries are built
in memory during step 1, but only WRITTEN to `references.bib` in step 3,
after OCR + enrich have both succeeded. Before that fix, the old code wrote
bib entries before running OCR at all, so all 4 recorded `download_arxiv`
runs left bib entries for papers that failed to ingest. Do not reorder this.

`PROJECT_ROOT` and `arxiv` are bound as plain module attributes (not
re-derived from `paper_fetch_mcp.config` inline) so tests can
`monkeypatch.setattr(core.ingest, "PROJECT_ROOT", tmp_path)` /
`monkeypatch.setattr(core.ingest, "arxiv", FakeArxivModule)` to run the
whole pipeline against a throwaway directory and a fake ArXiv client --
this pipeline must never be executed against the real corpus in a test.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import arxiv

from paper_fetch_mcp.config import CORPUS_ROOT as PROJECT_ROOT
from paper_fetch_mcp.core import jobs as jobs_mod

logger = logging.getLogger(__name__)

_MAX_IDS_PER_REQUEST = 10

# The `core.jobs` `kind` shared by both the `ingest_arxiv` tool and the
# `download_arxiv` alias -- both enqueue the exact same job kind/handler
# (spec §4 non-goal: old tool names stay as thin aliases over the new
# behaviour, they don't get a separate code path).
INGEST_JOB_KIND = "ingest_arxiv"


def download_arxiv_sync(ids: List[str]) -> Dict[str, Any]:
    """Run the full synchronous ArXiv ingest pipeline for `ids`.

    Returns a dict shaped either:
        {"ok": True, "results": {citation_key: content_or_status, ...}}
    or, on a total failure before any per-paper work could start:
        {"ok": False, "error": "<message>"}

    Never raises -- every failure path is caught and reported in the
    returned dict, matching the old monolith's behaviour (which always
    returned a YAML envelope, never propagated an exception to the MCP
    transport).
    """
    ids = ids[:_MAX_IDS_PER_REQUEST]
    results: Dict[str, str] = {}
    pre_proc_dir = PROJECT_ROOT / "pre_processing_v2"
    save_dir = pre_proc_dir / "new_papers"
    save_dir.mkdir(parents=True, exist_ok=True)
    bib_path = PROJECT_ROOT / "references.bib"

    downloaded_files = []

    # 1. Download and build BibTeX entries IN MEMORY ONLY. Do not write them
    # to references.bib yet -- that only happens after OCR/enrich succeeds
    # below (0.8: all 4 recorded download_arxiv runs left bib entries for
    # papers that failed to ingest, because the old code wrote the bib file
    # before running OCR at all).
    try:
        search = arxiv.Search(id_list=ids)
        new_entries = []

        for paper in search.results():
            try:
                # Generate key: firstAuthor + Year + firstWordTitle
                authors = paper.authors
                first_author = authors[0].name.split()[-1] if authors else "Unknown"
                first_author = "".join(c for c in first_author if c.isalnum())
                year = paper.published.year
                title_words = [w for w in paper.title.split() if w.isalnum()]
                first_word = title_words[0] if title_words else "Paper"
                first_word = "".join(c for c in first_word if c.isalnum())

                citation_key = f"{first_author}{year}{first_word}"
                filename = f"{citation_key}.pdf"
                path = save_dir / filename

                # Download (even if exists, we might want to ensure it's there for processing)
                if not path.exists():
                    paper.download_pdf(dirpath=str(save_dir), filename=filename)

                downloaded_files.append((citation_key, filename))

                # Construct BibTeX (not written yet -- see step 3 below)
                bib_entry = f"""
@article{{{citation_key},
    title = {{{paper.title}}},
    author = {{{' and '.join(a.name for a in paper.authors)}}},
    year = {{{year}}},
    journal = {{ArXiv Preprint}},
    url = {{{paper.entry_id}}},
    file = {{{citation_key}}},
    citation_key = {{{citation_key}}}
}}
"""
                new_entries.append(bib_entry)

            except Exception as e:
                results[paper.get_short_id()] = f"Failed download/setup: {e}"

    except Exception as e:
        return {"ok": False, "error": f"Arxiv bulk download failed: {e}"}

    # 2. Synchronous OCR Processing
    # We want to return content immediately.
    # Run OCR -> Enrich -> (only then) append BibTeX -> Read Content

    ocr_script = pre_proc_dir / "mistral_ocr.py"
    enrich_script = pre_proc_dir / "enrich_metadata.py"
    mistral_output_dir = pre_proc_dir / "mistral_output"

    try:
        # Run OCR (for all new papers in save_dir, simpler than selective for now)
        subprocess.run(
            ["uv", "run", "python", str(ocr_script), "--pdf-directory", str(save_dir)],
            check=True,
            capture_output=True,
        )

        # Run Enrich
        subprocess.run(
            ["uv", "run", "python", str(enrich_script)], check=True, capture_output=True
        )

    except subprocess.CalledProcessError as e:
        # Surface the failing script's own stderr verbatim (spec §11 /
        # finding F9: "fix [of enrich_metadata.py's exit 1] tracked
        # separately" -- this module does not attempt to fix that script,
        # only to report its failure honestly instead of a generic message
        # that gives the next person nothing to diagnose from).
        stderr_text = e.stderr
        if isinstance(stderr_text, bytes):
            stderr_text = stderr_text.decode("utf-8", errors="replace")
        stderr_text = (stderr_text or "").strip()
        logger.error("Sync processing failed: %s\nstderr:\n%s", e, stderr_text)
        message = (
            f"Processing failed during OCR/enrich step "
            f"(command={list(e.cmd)!r}, exit={e.returncode})."
        )
        if stderr_text:
            # Keep this bounded -- a runaway script's stderr should not by
            # itself blow the job record past output.respond()'s hard cap.
            message += f" stderr: {stderr_text[-4000:]}"
        return {"ok": False, "error": message}

    # 3. OCR/enrich succeeded -- NOW it is safe to append BibTeX entries.
    if new_entries:
        try:
            with open(bib_path, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_entries))
            logger.info("Added %d entries to references.bib", len(new_entries))
        except Exception as e:
            logger.error("Failed to append bibtex entries: %s", e)

    # 4. Read Content
    for ckey, _ in downloaded_files:
        # File should be at mistral_output/{ckey}/{ckey}.md
        md_path = mistral_output_dir / ckey / f"{ckey}.md"
        if md_path.exists():
            results[ckey] = md_path.read_text(encoding="utf-8")
        else:
            results[ckey] = "Processing failed: Output file not found."

    # 5. Trigger Background Pipeline (DB + Index)
    # We use Popen (detached) so we don't wait for Index building.
    try:
        pipeline_script = pre_proc_dir / "process_pipeline.py"
        # Since we already ran OCR/Enrich, we could use flags to skip, but
        # re-running is safe (just skips). Ensures consistency in DB/Index
        # based on bib/output.
        subprocess.Popen(
            ["uv", "run", "python", str(pipeline_script)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # Detach process
        )
        logger.info("Background pipeline triggered.")
    except Exception as e:
        logger.error("Failed to trigger background pipeline: %s", e)

    return {"ok": True, "results": results}


# ---------------------------------------------------------------------------
# Job-model wrapper (spec §6, §8 Phase 2). `ingest_arxiv` and the
# `download_arxiv` alias both call `enqueue_ingest_job`; `core.jobs`'s
# worker thread calls `ingest_job_handler` for every queued job of
# `INGEST_JOB_KIND`.
# ---------------------------------------------------------------------------


def enqueue_ingest_job(ids: List[str]) -> Dict[str, Any]:
    """Queue an ArXiv ingest job for `ids` and return
    `{"job_id": ..., "status": "queued"}` immediately.

    Never touches the network or the filesystem itself -- it only inserts a
    `core.jobs` row. The worker thread (`core.jobs.start_worker()`, started
    from the FastMCP lifespan) drains it in the background by calling
    `ingest_job_handler` below. IDs are capped the same way
    `download_arxiv_sync` caps them, so `job_status` and the actual pipeline
    run never disagree about how many IDs were accepted.
    """
    capped_ids = list(ids)[:_MAX_IDS_PER_REQUEST]
    job_id = jobs_mod.new_job(INGEST_JOB_KIND, {"ids": capped_ids})
    return {"job_id": job_id, "status": "queued"}


def ingest_job_handler(payload: Dict[str, Any]) -> Dict[str, Any]:
    """`core.jobs` worker entry point for `kind=INGEST_JOB_KIND` jobs.

    Runs the real synchronous pipeline (`download_arxiv_sync`) and RAISES
    `RuntimeError(outcome["error"])` on a top-level pipeline failure --
    `download_arxiv_sync` itself never raises (it always returns a dict),
    so this is the translation point that turns its `{"ok": False, "error":
    ...}` into an exception the worker catches and records verbatim as the
    job's `error` column (spec §11 / finding F9). On success, returns
    `outcome["results"]` (the per-citation-key content/status map) as the
    job's `result`.
    """
    ids = payload.get("ids") or []
    outcome = download_arxiv_sync(ids)
    if not outcome.get("ok", False):
        raise RuntimeError(outcome.get("error", "ingest_arxiv failed"))
    return outcome["results"]
