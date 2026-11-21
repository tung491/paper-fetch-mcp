"""Admin tools: server_status, append_bibtex_entries, symlink_references_bib.

Thin `@mcp.tool` wrappers. The two BibTeX tools keep their small amount of
file/bibtex-parsing logic inline rather than in a `core/*` module.

`server_status` reports the papers DB, the job queue, the last telemetry
row and the exit-node route -- the same components `/health` probes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import bibtexparser
from bibtexparser.bibdatabase import BibDatabase
from bibtexparser.bwriter import BibTexWriter

from paper_fetch_mcp import mcp
from paper_fetch_mcp.config import BIB_PATH, DB_PATH
from paper_fetch_mcp.core import exit_node
from paper_fetch_mcp.core import jobs as jobs_mod
from paper_fetch_mcp.core import output
from paper_fetch_mcp.core import papers
from paper_fetch_mcp.core import telemetry

logger = logging.getLogger(__name__)


@mcp.tool(timeout=45)
def server_status() -> str:
    """
    Check server health: the papers database, the job queue, the most
    recent telemetry row, and whether the exit-node route `get_paper` falls
    back to (Tailscale exit node via the sidecar proxy, see
    core/exit_node.py) is currently usable.

    Returns:
        YAML with a status block per component.
    """
    status: dict = {"server": "PaperFetchServer"}

    status["database"] = {"path": str(DB_PATH), "exists": DB_PATH.exists()}
    if DB_PATH.exists():
        try:
            with papers.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                status["database"]["paper_count"] = cursor.fetchone()[0]
                status["database"]["status"] = "connected"
        except Exception as e:
            status["database"]["status"] = f"error: {e}"
    else:
        status["database"]["status"] = "not_found"

    try:
        recent_jobs = jobs_mod.list_jobs(limit=20)
        status["jobs"] = {
            "recent": len(recent_jobs),
            "queued": sum(1 for j in recent_jobs if j.get("status") == "queued"),
            "running": sum(1 for j in recent_jobs if j.get("status") == "running"),
        }
    except Exception as e:
        status["jobs"] = {"status": f"error: {e}"}

    try:
        last = telemetry.last_call_row()
        status["telemetry"] = {"last_call": last}
    except Exception as e:
        status["telemetry"] = {"status": f"error: {e}"}

    try:
        status["exit_node"] = exit_node.status()
    except Exception as e:
        status["exit_node"] = {"status": f"error: {e}"}

    return output.respond(status)


@mcp.tool(timeout=45)
def append_bibtex_entries(citation_keys: List[str], reference_path: str) -> str:
    """
    Append specific BibTeX entries from the main references.bib to a
    destination file.

    Args:
        citation_keys: List of citation keys to extract.
        reference_path: Path to the destination .bib file.

    Returns:
        YAML with status, appended count, and any missing keys.
    """
    try:
        if not BIB_PATH.exists():
            return output.error("Main references.bib not found.")

        with open(BIB_PATH, "r") as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)

        entries_to_append = []
        found_keys = set()
        for entry in bib_database.entries:
            if entry.get("ID") in citation_keys:
                entries_to_append.append(entry)
                found_keys.add(entry.get("ID"))

        if not entries_to_append:
            return output.respond(
                {
                    "warning": "No matching entries found.",
                    "found_keys": [],
                    "missing_keys": citation_keys,
                }
            )

        db = BibDatabase()
        db.entries = entries_to_append
        writer = BibTexWriter()
        bibtex_str = writer.write(db)

        target_path = Path(reference_path)
        if not target_path.parent.exists():
            return output.error(f"Parent directory of {target_path} does not exist.")

        with open(target_path, "a", encoding="utf-8") as f:
            f.write("\n" + bibtex_str)

        return output.respond(
            {
                "status": "success",
                "appended_count": len(entries_to_append),
                "found_keys": list(found_keys),
                "missing_keys": list(set(citation_keys) - found_keys),
            }
        )
    except Exception as e:
        return output.error(f"Failed to append bibtex entries: {e}")


@mcp.tool(timeout=45)
def symlink_references_bib(source_dir: str, target_dir: str) -> str:
    """
    Create a symbolic link for references.bib from source_dir to target_dir.
    Useful for sharing the references file across different workspace
    stages.

    Args:
        source_dir: Directory containing the source references.bib.
        target_dir: Directory where the symlink should be created.

    Returns:
        YAML with status and symlink path.
    """
    try:
        source_path = Path(source_dir) / "references.bib"
        target_path = Path(target_dir) / "references.bib"

        if not source_path.exists():
            return output.error(f"Source file {source_path} does not exist.")

        if not Path(target_dir).exists():
            return output.error(f"Target directory {target_dir} does not exist.")

        if target_path.exists():
            if target_path.is_symlink() and target_path.resolve() == source_path.resolve():
                return output.respond({"status": "exists", "path": str(target_path)})
            return output.error(f"Target file {target_path} already exists.")

        os.symlink(source_path, target_path)
        return output.respond({"status": "created", "path": str(target_path)})
    except Exception as e:
        return output.error(f"Failed to create symlink: {e}")
