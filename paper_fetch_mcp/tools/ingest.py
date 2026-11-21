"""Ingestion tools: ingest_arxiv (async job), job_status, and download_arxiv
(thin alias).

Design spec `docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md`,
§6 "Tool surface", §8 Phase 2, finding F9. All three tools are thin
`@mcp.tool` wrappers -- the real pipeline lives in `core/ingest.py`, the job
table/worker in `core/jobs.py`; nothing here runs a subprocess or touches
the network directly.

Phase 1 (a prior wave) restored `download_arxiv` to a fully synchronous
call into `core.ingest.download_arxiv_sync` after an earlier cut enqueued a
job that nothing drained -- see `core/ingest.py`'s module docstring for
that history. Phase 2 (this wave) makes the job model real: `core.jobs` now
has a worker thread (`core.jobs.start_worker()`, started from the FastMCP
lifespan in `app.py`), so it is safe to go back to enqueue-and-return-
immediately here. `ingest_arxiv` is the new primary name; `download_arxiv`
stays registered (spec §4 non-goal: "old names stay as thin aliases") with
its exact original signature (`ids: List[str]`) -- pinned by
`tests/golden/tools_list.json` -- but now returns the job record instead of
blocking for the pipeline's result.
"""

from __future__ import annotations

import logging
from typing import List

from paper_fetch_mcp import mcp
from paper_fetch_mcp.core import ingest as ingest_mod
from paper_fetch_mcp.core import jobs as jobs_mod
from paper_fetch_mcp.core import output

logger = logging.getLogger(__name__)

# Register the worker-side handler for kind="ingest_arxiv" jobs. Import-time
# side effect, same pattern as the @mcp.tool registrations below -- this
# module (like every tools/*.py module) is only ever imported from
# paper_fetch_mcp.app.build(), never at package import time, so this never
# runs merely from `import paper_fetch_mcp`.
jobs_mod.register_handler(ingest_mod.INGEST_JOB_KIND, ingest_mod.ingest_job_handler)


@mcp.tool(timeout=45)
def ingest_arxiv(ids: List[str]) -> str:
    """
    Queue an ArXiv ingest job (download -> Mistral OCR -> enrich -> DB/index
    update) and return immediately with a job id. Poll job_status(job_id)
    for progress and the eventual result or error.

    Args:
        ids: List of ArXiv IDs (e.g., ["2310.12345"]). Maximum 10 per
            request; extra ids beyond the 10th are dropped.

    Returns:
        YAML envelope with `job_id` and `status: "queued"`.
    """
    outcome = ingest_mod.enqueue_ingest_job(ids)
    return output.respond(outcome)


@mcp.tool(timeout=45)
def job_status(job_id: str) -> str:
    """
    Poll the status of a background job created by ingest_arxiv (or its
    download_arxiv alias).

    Args:
        job_id: The job id returned by ingest_arxiv/download_arxiv.

    Returns:
        YAML envelope with the job record: `status` ("queued", "running",
        "done", or "error"), `result` (on success), `error` (verbatim
        failure message, on failure), and timestamps. `found: false` if
        `job_id` is unknown.
    """
    record = jobs_mod.get_job(job_id)
    if record is None:
        return output.respond({"found": False, "job_id": job_id})
    return output.respond(record)


@mcp.tool(timeout=45)
def download_arxiv(ids: List[str]) -> str:
    """
    Alias for ingest_arxiv (spec §4 non-goal: old tool names stay as thin
    aliases). Queues the same background ingest job and returns the job
    record immediately -- it no longer blocks for OCR'd content. Poll
    job_status(job_id) (the returned record's `job_id`) for the outcome.

    Args:
        ids: List of ArXiv IDs (e.g., ["2310.12345"]). Maximum 10 per
            request; extra ids beyond the 10th are dropped.

    Returns:
        YAML envelope with the job record (job_id, status, ...).
    """
    outcome = ingest_mod.enqueue_ingest_job(ids)
    record = jobs_mod.get_job(outcome["job_id"]) or outcome
    return output.respond(record)
