# Setup tutorial for AI agents

This is a step-by-step runbook for an AI coding agent (Claude Code, Codex,
Gemini CLI, or similar) to install this MCP server and register it with a
client. Run the steps in order. Every step ends with a check; do not continue
past a check that fails.

Conventions used below:

- `$R` is the absolute path of this repository's checkout.
- The server is always registered under the name **`agentic-rag-mcp`**, so its
  tools are addressed as `mcp__agentic-rag-mcp__<tool>`. Do not invent another
  name.
- Nothing here needs `sudo`.

## What to ask the human before starting

Only these three things cannot be discovered from the machine. Ask once, then
proceed.

1. **Corpus path.** Where is the `ai_researcher` checkout that holds the paper
   corpus? If there is none, the server still works for online papers; skip
   step 3's `CORPUS_ROOT` line.
2. **Semantic Scholar API key.** Optional. Without it Semantic Scholar calls
   work but are slower and rate-limited.
3. **Which client(s)** to register with, and whether to use stdio (one server
   process per client session) or the shared HTTP service (step 6).

## Step 1 — Check prerequisites

```bash
uv --version          # required
git --version         # required
which chromium chromium-browser google-chrome google-chrome-stable   # optional
```

- `uv` is required; it installs Python 3.13 itself if missing. If `uv` is
  absent, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`
  after confirming with the human.
- A Chromium or Chrome binary is optional. With one on `PATH`, `get_paper` and
  `fetch_publisher_html` can render publisher pages that block plain HTTP
  clients. Without one those pages fall back to the Semantic Scholar abstract.

## Step 2 — Get the code and install dependencies

```bash
git clone git@github.com:tung491/paper-fetch-mcp.git
cd paper-fetch-mcp
R="$(pwd)"
uv sync
```

Check:

```bash
"$R/.venv/bin/python" -c "import paper_fetch_mcp, fastmcp; print('ok', fastmcp.__version__)"
```

Expected: `ok 3.x.y`. The major version must be 3.

## Step 3 — Configure

```bash
cp "$R/paper_fetch_mcp/.env.example" "$R/paper_fetch_mcp/.env"
```

Edit `$R/paper_fetch_mcp/.env`. It is gitignored; never commit it.

| Line | What to put |
|------|-------------|
| `CORPUS_ROOT=` | Absolute path of the `ai_researcher` checkout. Delete the line if there is no corpus |
| `SEMANTIC_SCHOLAR_API_KEY=` | The key, or delete the line |
| `MISTRAL_API_KEY=` | Only needed for `ingest_arxiv`. Delete the line otherwise |

Remove every placeholder value (`your-...-here`, `/home/you/...`). A
placeholder left in place is sent as a real key or used as a real path.

Check that the corpus path is right (skip if there is no corpus):

```bash
C="$(grep '^CORPUS_ROOT=' "$R/paper_fetch_mcp/.env" | cut -d= -f2)"
ls "$C/references.bib" "$C/pre_processing_v2/papers.db" "$C/pre_processing_v2/mistral_output" >/dev/null && echo corpus-ok
```

If `papers.db` is missing but `references.bib` and `mistral_output/` exist,
build it once from inside the corpus checkout:

```bash
(cd "$C" && uv run python pre_processing_v2/build_database.py)
```

## Step 4 — Verify the server before registering it

Run the unit tests (about 25 seconds, no network, no server process):

```bash
cd "$R" && uv run pytest -q
```

Expected: all tests pass, one skipped.

Then start the server over stdio the same way a client will, list its tools
and call `server_status`:

```bash
cd "$R" && .venv/bin/python - <<'EOF' 2>/dev/null
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

async def main():
    transport = StdioTransport(command=".venv/bin/python", args=["-m", "paper_fetch_mcp"])
    async with Client(transport) as client:
        names = sorted(t.name for t in await client.list_tools())
        print(len(names), "tools:", ", ".join(names))
        print((await client.call_tool("server_status", {})).data)

asyncio.run(main())
EOF
```

Expected:

- `17 tools:` followed by the tool names. `hybrid_search`, `hyde_query`,
  `stepback_query`, `list_available_tags` and `reload_indices` must **not**
  appear; this server has no search over paper text.
- `database.status: connected` and a `paper_count` above zero when
  `CORPUS_ROOT` is set. `status: not_found` means `CORPUS_ROOT` is wrong or
  unset.

## Step 5 — Register with the client (stdio)

Each command launches the repository's own virtualenv Python with `PYTHONPATH`
set, so it works from any working directory. Use the absolute path for `$R`.

Claude Code:

```bash
claude mcp add -s user agentic-rag-mcp -e PYTHONPATH=$R -- $R/.venv/bin/python -m paper_fetch_mcp
claude mcp list | grep agentic-rag-mcp
```

Codex:

```bash
codex mcp add agentic-rag-mcp --env PYTHONPATH=$R -- $R/.venv/bin/python -m paper_fetch_mcp
```

Then add under `[mcp_servers.agentic-rag-mcp]` in `~/.codex/config.toml`:

```toml
tool_timeout_sec = 120
```

Gemini CLI, in `~/.gemini/settings.json` (merge into an existing `mcpServers`
object; do not overwrite the file):

```json
{
  "mcpServers": {
    "agentic-rag-mcp": {
      "command": "/ABSOLUTE/PATH/TO/paper-fetch-mcp/.venv/bin/python",
      "args": ["-m", "paper_fetch_mcp"],
      "env": { "PYTHONPATH": "/ABSOLUTE/PATH/TO/paper-fetch-mcp" }
    }
  }
}
```

Any other MCP client: command `$R/.venv/bin/python`, arguments
`-m paper_fetch_mcp`, environment `PYTHONPATH=$R`, server name
`agentic-rag-mcp`.

Rules for every client:

- If a server named `agentic-rag-mcp` is already registered, stop and ask the
  human before replacing it.
- Set the tool timeout to at least 90 seconds. `get_paper` and
  `fetch_publisher_html` can take that long on an online fetch.
- The client must be restarted, or a new session started, before the tools
  appear.

## Step 6 — Alternative: one shared HTTP service

Use this instead of step 5 when several clients or machines should share one
server process.

```bash
ss -ltnp | grep ':8766 '        # must print nothing; if it does, see Troubleshooting
"$R/paper_fetch_mcp/run_server.sh" start
curl -s http://127.0.0.1:8766/health
```

Expected: `run_server.sh` prints `Server started`, and `/health` returns JSON
with `"status":"ok"`.

Register the URL instead of a command:

```bash
claude mcp add -s user --transport http agentic-rag-mcp http://127.0.0.1:8766/mcp
```

Manage the service with `run_server.sh {status|logs|restart|stop}`. Stop it
with `run_server.sh stop` when it is no longer needed; do not leave a second
copy running.

## Step 7 — Confirm from inside the client

In a new client session, make these three calls:

1. `server_status` — expect `server: PaperFetchServer`.
2. `get_paper` with `identifier: "arXiv:1706.03762"` and
   `max_chars_per_paper: 500` — expect `found: true`, `source: online`.
3. With a corpus only: `search_metadata` with `query: "semantic communication"`
   and `limit: 3` — expect a non-empty `results` list.

Setup is complete when all applicable calls succeed.

## How to use the tools once set up

- To find papers: `search_metadata` (local corpus metadata) or
  `semantic_scholar_search` (online). There is no full-text or semantic search.
- To read a paper: `get_paper` with a citation key, DOI, arXiv id, URL or
  title. Page through a long paper with `offset`, or pass `save_path` to write
  the whole paper to disk in one call.
- To add an arXiv paper to the corpus: `ingest_arxiv`, then poll `job_status`.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `ModuleNotFoundError: No module named 'paper_fetch_mcp'` | `PYTHONPATH` is not set to `$R` in the client registration |
| `ModuleNotFoundError: bibtexparser.bibdatabase` | Dependencies were installed without the lock file. Run `uv sync` in `$R` |
| `database.status: not_found`, or every corpus tool reports a miss | `CORPUS_ROOT` is unset or wrong, or `papers.db` has not been built (step 3) |
| Tools do not appear in the client | Restart the client; then check `claude mcp list` (or the client's equivalent) for a connection error |
| A tool call times out near 30 or 60 seconds | Client tool timeout is too low; raise it to at least 90 seconds |
| `get_paper` returns `partial: true` with only an abstract | The publisher blocked the fetch. Install Chromium (step 1) |
| Port 8766 is already in use | Run `run_server.sh status`. If it is this server, reuse it. If it is another process, ask the human; do not kill it and do not pick a different port |
| `semantic_scholar_search` is slow or returns an upstream error | Unauthenticated rate limit. Set `SEMANTIC_SCHOLAR_API_KEY` |

## Safety rules for the agent

- Never commit `.env` or print its contents.
- Never stop or reconfigure a process you did not start. On a shared host
  another server (for example `agentic_rag_mcp` on port 8765) may be in use by
  other people.
- `ingest_arxiv` writes into the corpus checkout and appends to its
  `references.bib`. Only call it when the human asks for a paper to be added.
