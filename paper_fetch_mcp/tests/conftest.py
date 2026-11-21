"""Shared pytest configuration for paper_fetch_mcp/tests.

Hermetic-by-default network routing: `core.exit_node` probes a local proxy
port (and may shell out to `tailscale --socket ... status`) whenever a
fetch fails for an access reason, and `tools/content.get_paper` now falls
through to a real network fetch on a corpus miss. Neither may happen as a
side effect of an unrelated unit test -- on the dev box the exit-node
sidecar IS usually running, so a test that merely simulates a connection
error would otherwise start routing through the real exit node.

This autouse fixture therefore turns both off for every test; tests that
exercise the online fallback or the exit-node route opt back in by
monkeypatching `paper_fetch_mcp.config.EXIT_NODE_MODE` /
`GET_PAPER_ONLINE_FALLBACK` (and stubbing `exit_node.availability`) as
`tests/test_core_exit_node.py`, `tests/test_core_online.py` and the
publisher/content tests do.

Both the environment variables AND the already-imported `config` module's
attributes are patched: `core.exit_node`/`core.online` read
`config.<NAME>` at call time, while `tests/test_tool_contract.py` purges and
re-imports the whole package (re-reading the env), so either path alone
would leave a gap.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _offline_network_routing(monkeypatch):
    monkeypatch.setenv("EXIT_NODE_MODE", "off")
    monkeypatch.setenv("GET_PAPER_ONLINE_FALLBACK", "0")
    # Never launch a real Chromium from core.publisher's own-network step;
    # tests that exercise it patch `_headless_browser_bin` back on.
    monkeypatch.setenv("PUBLISHER_HEADLESS_BROWSER", "0")
    cfg =sys.modules.get("paper_fetch_mcp.config")
    if cfg is not None:
        monkeypatch.setattr(cfg, "EXIT_NODE_MODE", "off", raising=False)
        monkeypatch.setattr(cfg, "GET_PAPER_ONLINE_FALLBACK", False, raising=False)
    # Never let a probe cached by an earlier (opted-in) test leak forward.
    exit_node = sys.modules.get("paper_fetch_mcp.core.exit_node")
    if exit_node is not None:
        monkeypatch.setattr(exit_node, "_probe_cache", (0.0, None), raising=False)
    yield
