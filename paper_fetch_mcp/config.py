"""Centralized configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the package directory (if present)
_pkg_dir = Path(__file__).resolve().parent
load_dotenv(_pkg_dir / ".env", override=False)

# --- Paths ---
# REPO_ROOT is this repository. CORPUS_ROOT is the directory that holds the
# paper corpus this server reads (and that `ingest_arxiv` writes into):
#
#   <CORPUS_ROOT>/references.bib
#   <CORPUS_ROOT>/pre_processing_v2/papers.db
#   <CORPUS_ROOT>/pre_processing_v2/mistral_output/<paper id>/...
#   <CORPUS_ROOT>/pre_processing_v2/{mistral_ocr,enrich_metadata,process_pipeline}.py
#
# The corpus is not part of this repository, so set CORPUS_ROOT (in `.env` or
# the environment) to the checkout that owns it. Left unset it falls back to
# REPO_ROOT, where every corpus-backed tool reports a clean miss / "not found"
# and the online tools still work.
REPO_ROOT = _pkg_dir.parent
CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "").strip() or REPO_ROOT).expanduser()
PRE_PROCESSING_DIR = CORPUS_ROOT / "pre_processing_v2"
DB_PATH = PRE_PROCESSING_DIR / "papers.db"
MISTRAL_OUTPUT_DIR = PRE_PROCESSING_DIR / "mistral_output"
BIB_PATH = CORPUS_ROOT / "references.bib"

# Server-owned state (job queue, call telemetry). Kept in this repository
# rather than next to the corpus so a second server reading the same corpus
# never shares -- or reconciles away -- this server's job queue.
STATE_DIR = Path(os.environ.get("STATE_DIR", str(REPO_ROOT / ".state"))).expanduser()

# --- Semantic Scholar ---
# Optional: the Graph API works unauthenticated at a lower rate limit, so an
# empty value here is a valid ("no key configured") state, not an error.
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# --- Mistral OCR (Azure-hosted) ---
# Required to call the OCR endpoint; callers must check for "" themselves and
# raise a clear error rather than send an empty Authorization header.
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# --- Defaults ---
MAX_RESULTS_LIMIT = 200
DEFAULT_MAX_RESULTS = 50

# --- Exit-node route for online paper access (core/exit_node.py) ---
# When `get_paper` misses the local corpus it fetches the paper online. If
# the fetch fails for an ACCESS reason from this host's own network (paywall
# 403, geo-block 451, connection refused/timed out, ...) it is retried once
# through a Tailscale exit node -- by default `ubuntu-lab`, which sits on
# the lab network with institutional publisher access.
#
# This host cannot simply `tailscale set --exit-node=...`: it advertises an
# exit node itself (Tailscale refuses to both advertise and use one) and its
# tailscaled runs in userspace-networking mode (no TUN, so a host-level exit
# node would not even affect plain sockets). Instead a SECOND userspace
# tailscaled -- the "exit-node sidecar", which this repository does not ship
# -- joins the tailnet as its own node, uses EXIT_NODE_NAME as *its* exit
# node, and exposes a local SOCKS5 proxy. Requests that need the exit node
# go through that proxy; everything else on the host is untouched.
#
#   EXIT_NODE_MODE   auto  -> use the proxy whenever the sidecar is up and
#                             actually has its exit node engaged (default)
#                    proxy -> same as auto (explicit)
#                    off   -> never route via the exit node
#   EXIT_NODE_NAME             tailnet hostname of the exit node
#   EXIT_NODE_PROXY_URL        the sidecar's SOCKS5 (socks5h:// so DNS is
#                              resolved at the exit side, not locally)
#   EXIT_NODE_TAILSCALE_SOCKET the sidecar's tailscaled control socket; used
#                              only to CHECK that its exit node is engaged
EXIT_NODE_NAME = os.environ.get("EXIT_NODE_NAME", "ubuntu-lab").strip()
EXIT_NODE_PROXY_URL = os.environ.get("EXIT_NODE_PROXY_URL", "socks5h://127.0.0.1:1056").strip()
EXIT_NODE_TAILSCALE_SOCKET = os.environ.get(
    "EXIT_NODE_TAILSCALE_SOCKET", "/var/run/tailscale-exit/tailscaled.sock"
).strip()
_EXIT_NODE_MODE_RAW = os.environ.get("EXIT_NODE_MODE", "auto").strip().lower()
EXIT_NODE_MODE = _EXIT_NODE_MODE_RAW if _EXIT_NODE_MODE_RAW in ("auto", "proxy", "off") else "auto"

# --- get_paper online fallback (core/online.py) ---
# "1" (default): a `get_paper` miss on the local corpus falls through to an
# online fetch (publisher page for a DOI/URL, arxiv.org for an arXiv id,
# Semantic Scholar to resolve a bare title first), with the exit-node retry
# above. "0": `get_paper` stays purely local, as before.
GET_PAPER_ONLINE_FALLBACK = os.environ.get("GET_PAPER_ONLINE_FALLBACK", "1").strip() != "0"

# --- get_paper save_path (core/papers.save_paper_content) ---
# `get_paper(..., save_path=...)` writes the paper's FULL assembled content
# (not just the returned window) to disk. A RELATIVE save_path is resolved
# under this directory rather than against the server process's cwd -- the
# MCP server's cwd is whatever launched it, so a bare "smith2024.md" from an
# agent would otherwise land somewhere unpredictable. An absolute path is
# used as given.
PAPER_SAVE_DIR = Path(
    os.environ.get("PAPER_SAVE_DIR", str(REPO_ROOT / "saved_papers"))
).expanduser()

# --- Publisher asset cache (core/images.py) ---
# Figures and full-text PDFs fetched together with an online paper (e.g. from
# inside the headless-browser session that passed a publisher's bot
# challenge, where a plain GET of the file would be refused) are kept here,
# keyed by URL, so a later `get_paper(..., save_path=...)` can copy them next
# to the saved file without fetching them again.
PUBLISHER_ASSET_CACHE_DIR = Path(
    os.environ.get(
        "PUBLISHER_ASSET_CACHE_DIR", str(REPO_ROOT / ".cache" / "publisher_assets")
    )
).expanduser()
