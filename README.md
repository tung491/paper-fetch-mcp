# paper-fetch-mcp

MCP server that gives AI agents full-text access to research papers: lookup in a
local paper corpus, online fetch from publishers and arXiv when a paper is not
there, plus Semantic Scholar and alphaXiv lookups, arXiv ingestion and BibTeX
helpers.

It is derived from `agentic_rag_mcp` (in the `ai_researcher` repository) with
all retrieval over paper text removed: there is no FAISS vector index, no BM25
index, no HyDE or step-back query rewriting, and no Ollama dependency. The
tools `hybrid_search`, `hyde_query`, `stepback_query`, `list_available_tags`
and `reload_indices` and the `corpus://tags` resource do not exist here. Agents
find papers with `search_metadata` (local metadata) or `semantic_scholar_search`
(online) and read them with `get_paper`.

The server starts in well under a second, because nothing is loaded into memory
at startup.

## Architecture

One entrypoint, `python -m paper_fetch_mcp`, builds the FastMCP app (see
`app.py` / `__main__.py`) from a `core/` + `tools/` split: `core/` holds the
implementation (no `@mcp.tool` decorators, no I/O at import time — all startup
work happens inside `app.build()`'s lifespan), and `tools/` holds thin
`@mcp.tool` wrappers that parse arguments and format the response via
`core/output.py`.

```
paper_fetch_mcp/
├── __init__.py        # Shared FastMCP instance
├── __main__.py        # `python -m paper_fetch_mcp` CLI entrypoint
├── app.py             # build() -- lifespan, middleware, tool registration, /health
├── config.py          # Environment-based configuration
├── core/              # papers, online, publisher, images, exit_node, s2, alphaxiv,
│                      #   ingest, jobs, output, telemetry, text
├── tools/
│   ├── admin.py       # server_status, append_bibtex_entries, symlink_references_bib
│   ├── content.py     # get_paper, doi2content, get_content_by_citation_key,
│   │                  #   list_figures, get_image_path
│   ├── external.py    # semantic_scholar_search, semantic_scholar_paper,
│   │                  #   alphaxiv_paper_lookup, alphaxiv_batch_lookup, fetch_publisher_html
│   ├── ingest.py      # ingest_arxiv, download_arxiv, job_status
│   └── search.py      # search_metadata
├── tests/             # pytest test suite
├── run_server.sh      # background streamable-http service
└── .env.example       # Template for .env (gitignored)
```

## The corpus lives outside this repository

The corpus-backed tools read a paper corpus that this repository does not
contain. `CORPUS_ROOT` points at the checkout that owns it (the `ai_researcher`
repository):

```
<CORPUS_ROOT>/references.bib
<CORPUS_ROOT>/pre_processing_v2/papers.db
<CORPUS_ROOT>/pre_processing_v2/mistral_output/<paper id>/...
<CORPUS_ROOT>/pre_processing_v2/{mistral_ocr,enrich_metadata,process_pipeline}.py
```

`ingest_arxiv` writes into that corpus: it downloads into
`pre_processing_v2/new_papers`, runs the corpus checkout's OCR and enrich
scripts with `uv run` from `CORPUS_ROOT`, appends to its `references.bib`, and
then triggers its `process_pipeline.py`.

Without `CORPUS_ROOT` the server still runs: corpus-backed tools report misses
and the online tools (`get_paper` for a DOI, arXiv id, URL or title,
`fetch_publisher_html` without caching, Semantic Scholar, alphaXiv) work
normally.

The server's own state — the job queue and call telemetry — is kept in
`.state/` in this repository, not next to the corpus, so it is never shared
with another server reading the same corpus.

## Configuration

```bash
cp paper_fetch_mcp/.env.example paper_fetch_mcp/.env
```

| Variable | Description |
|----------|-------------|
| `CORPUS_ROOT` | Checkout that owns the paper corpus (see above). Default: this repository, i.e. no corpus |
| `STATE_DIR` | Job queue and telemetry databases (default `<repo>/.state`) |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key (optional; unauthenticated calls work at a lower rate limit) |
| `MISTRAL_API_KEY` | Mistral OCR key, used by the corpus checkout's `pre_processing_v2/mistral_ocr.py`, which `ingest_arxiv` runs as a subprocess |
| `GET_PAPER_ONLINE_FALLBACK` | `1` (default): a `get_paper` corpus miss is fetched online; `0`: `get_paper` stays local-only |
| `PAPER_SAVE_DIR` | Base directory a relative `get_paper(save_path=...)` is resolved under (default `<repo>/saved_papers`) |
| `PUBLISHER_ASSET_CACHE_DIR` | Cache for figures and PDFs fetched with an online paper (default `<repo>/.cache/publisher_assets`) |
| `EXIT_NODE_MODE` | `auto` (default) / `off`: whether an online fetch may retry through the Tailscale exit node when this host's network cannot access a paper |
| `EXIT_NODE_NAME` | Exit node the sidecar uses (default `ubuntu-lab`) |
| `EXIT_NODE_PROXY_URL` | The exit-node sidecar's SOCKS5 proxy (default `socks5h://127.0.0.1:1056`) |
| `EXIT_NODE_TAILSCALE_SOCKET` | Sidecar tailscaled socket, used only to check its exit node is engaged (default `/var/run/tailscale-exit/tailscaled.sock`) |

### Exit-node sidecar

`get_paper` falls back to the network when a paper is not in the corpus. If the
direct fetch fails for an access reason (publisher 403, geo-block, connection
failure, or a paywall stub), it is retried once through the Tailscale exit node
named by `EXIT_NODE_NAME`, which has institutional publisher access. That route
is a second userspace tailscaled exposing a local SOCKS5 proxy at
`EXIT_NODE_PROXY_URL`. This repository does not ship that sidecar: run one
yourself, or leave it absent, in which case the server never uses the exit
node and `server_status` says why under `exit_node`.

## Running

```bash
uv sync

# stdio transport
uv run python -m paper_fetch_mcp

# streamable-http, as run_server.sh does
uv run python -m paper_fetch_mcp --transport streamable-http --host 0.0.0.0 --port 8766
```

`paper_fetch_mcp/run_server.sh {start|stop|restart|status|logs}` runs the
streamable-http server as a background service and polls `/health` before
reporting it up. The default port is 8766, one above `agentic_rag_mcp`'s 8765,
so both servers can run on the same host.

### Registering with agent CLIs (user scope)

A step-by-step runbook an AI agent can follow end to end, with a check after
every step, is in [docs/agent-setup.md](docs/agent-setup.md).

Register the server under the name `agentic-rag-mcp`, so agent clients and
prompts keep addressing its tools as `mcp__agentic-rag-mcp__<tool>`:

```bash
R=/path/to/paper-fetch-mcp
claude mcp add -s user agentic-rag-mcp -e PYTHONPATH=$R -- $R/.venv/bin/python -m paper_fetch_mcp
codex mcp add agentic-rag-mcp --env PYTHONPATH=$R -- $R/.venv/bin/python -m paper_fetch_mcp
```

`get_paper` and `fetch_publisher_html` can take up to 90 s on an online fetch,
so raise the client's tool timeout if it is lower than that.

## Tools (17 total)

### Content access
| Tool | Description |
|------|-------------|
| `get_paper(identifier, max_chars_per_paper, offset, save_path)` | Umbrella lookup by citation key, DOI, arXiv id or title with paging. On a corpus miss fetches the paper online, retrying via the Tailscale exit node when this network cannot access it (`source: online`, `route: direct \| exit_node \| cache`). With `save_path` the paper's full content is written to that file (or `<dir>/<citation_key>.md`) and reported in `saved: {path, chars, bytes, overwrote}`; the result itself stays windowed |
| `doi2content(doi)` | Retrieve corpus paper content by DOI |
| `get_content_by_citation_key(citation_keys, sections)` | Retrieve corpus paper sections by citation key |
| `list_figures(citation_key)` | Enumerate a corpus paper's figure files |
| `get_image_path(citation_key, figure_id)` | Locate a figure file |

### Search
| Tool | Description |
|------|-------------|
| `search_metadata(query, author, year_from, year_to, limit)` | Metadata search (title, abstract, author, year) over the papers database |

### External
| Tool | Description |
|------|-------------|
| `semantic_scholar_search(query, venue, year, fields_of_study, limit)` | Semantic Scholar search. `venue` must be the exact S2 venue name (e.g. `"IEEE Transactions on Wireless Communications"`); abbreviations return zero results. Rows carry `in_corpus` when the paper is already in the corpus |
| `semantic_scholar_paper(identifier)` | Single-paper Semantic Scholar lookup by DOI, arXiv id or S2 id |
| `alphaxiv_paper_lookup(paper_id_or_url, max_chars)` | Fetch a single alphaXiv paper overview |
| `alphaxiv_batch_lookup(paper_ids_or_urls, max_chars)` | Fetch several alphaXiv papers |
| `fetch_publisher_html(identifier, force_refresh)` | Fetch and cache publisher content for a DOI or URL; same direct-then-exit-node route as `get_paper` |

### Ingestion
| Tool | Description |
|------|-------------|
| `ingest_arxiv(ids)` | Queue a background job that downloads, OCRs and adds arXiv papers to the corpus |
| `download_arxiv(ids)` | Alias for `ingest_arxiv` |
| `job_status(job_id)` | Poll a background job |

### Admin
| Tool | Description |
|------|-------------|
| `server_status()` | Papers database, job queue, last telemetry row, exit-node route |
| `append_bibtex_entries(citation_keys, reference_path)` | Copy BibTeX entries from the corpus `references.bib` to another file |
| `symlink_references_bib(source_dir, target_dir)` | Symlink a `references.bib` between directories |

## Testing

```bash
uv run pytest

# opt-in: boots a real server on an ephemeral port and calls Semantic Scholar
PAPER_FETCH_SMOKE=1 uv run pytest paper_fetch_mcp/tests/test_smoke_http.py
```

The tool roster and input schemas are pinned in
`paper_fetch_mcp/tests/golden/tools_list.json`; re-bless a deliberate change
with `UPDATE_TOOL_CONTRACT=1 uv run pytest paper_fetch_mcp/tests/test_tool_contract.py`.
