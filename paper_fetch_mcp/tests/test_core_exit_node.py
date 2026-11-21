"""Tests for paper_fetch_mcp.core.exit_node -- the direct-then-exit-node GET
behind `get_paper`'s online fallback and `core.publisher`'s fetches.

All HTTP is stubbed by patching `requests.get` (the module-level function
`core.exit_node` calls), so the tests can assert on the `proxies=` keyword
that distinguishes the exit-node attempt from the direct one. The
availability probe (TCP connect + `tailscale status`) is stubbed too; no
test here opens a socket or runs a subprocess.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest
import requests

from paper_fetch_mcp import config
from paper_fetch_mcp.core import exit_node


def _resp(status: int) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = f"<html>{status}</html>"
    return r


@pytest.fixture
def exit_node_on(monkeypatch):
    """Opt back into the exit-node route (conftest turns it off) with the
    availability probe stubbed to 'available'."""
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_NAME", "tung-lab-pc")
    monkeypatch.setattr(config, "EXIT_NODE_PROXY_URL", "socks5h://127.0.0.1:1056")
    with patch.object(
        exit_node,
        "availability",
        return_value={
            "mode": "auto",
            "exit_node": "tung-lab-pc",
            "proxy_url": "socks5h://127.0.0.1:1056",
            "available": True,
            "reason": "stubbed",
        },
    ):
        yield


@pytest.fixture
def exit_node_unavailable(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    with patch.object(
        exit_node,
        "availability",
        return_value={
            "mode": "auto",
            "exit_node": "tung-lab-pc",
            "proxy_url": "socks5h://127.0.0.1:1056",
            "available": False,
            "reason": "exit-node proxy 127.0.0.1:1056 is not listening",
        },
    ):
        yield


# ---------------------------------------------------------------------------
# looks_like_access_failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        requests.ConnectionError("refused"),
        requests.exceptions.SSLError("handshake"),  # subclass of ConnectionError
        requests.Timeout("slow"),
        requests.exceptions.ReadTimeout("slow"),
    ],
)
def test_connection_level_errors_are_access_failures(exc):
    assert exit_node.looks_like_access_failure(exc=exc) is True


def test_other_exceptions_are_not_access_failures():
    assert exit_node.looks_like_access_failure(exc=requests.exceptions.InvalidURL("x")) is False
    assert exit_node.looks_like_access_failure(exc=ValueError("x")) is False


@pytest.mark.parametrize("status", [401, 403, 407, 429, 451])
def test_access_statuses(status):
    assert exit_node.looks_like_access_failure(response=_resp(status)) is True


@pytest.mark.parametrize("status", [200, 202, 302, 404, 410, 500, 502, 503])
def test_non_access_statuses(status):
    assert exit_node.looks_like_access_failure(response=_resp(status)) is False


def test_nothing_given_is_not_a_failure():
    assert exit_node.looks_like_access_failure() is False


# ---------------------------------------------------------------------------
# get(): direct path
# ---------------------------------------------------------------------------


def test_direct_success_never_touches_proxy(exit_node_on):
    ok = _resp(200)
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=ok) as g:
        r = exit_node.get("https://example.org/p", timeout=5)
    assert r.response is ok
    assert r.route == "direct"
    assert r.exit_node is None and r.direct_error is None
    assert g.call_count == 1
    assert "proxies" not in g.call_args.kwargs


def test_direct_404_is_returned_not_retried(exit_node_on):
    """A 404 means the resource is missing -- a different egress IP will not
    help, so no exit-node attempt is made."""
    nf = _resp(404)
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=nf) as g:
        r = exit_node.get("https://example.org/p", timeout=5)
    assert r.response is nf and r.route == "direct"
    assert g.call_count == 1


def test_direct_only_403_is_returned_not_retried(exit_node_on):
    """`direct_only=True`: even an access failure with the exit node up stays
    on this host's network -- the caller has more own-network attempts."""
    forbidden = _resp(403)
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=forbidden) as g, \
         patch("paper_fetch_mcp.core.exit_node.availability") as avail:
        r = exit_node.get("https://ieeexplore.ieee.org/document/1", timeout=5, direct_only=True)
    assert r.response is forbidden and r.route == "direct"
    assert r.direct_error is None and r.skipped_reason is None
    assert g.call_count == 1 and "proxies" not in g.call_args.kwargs
    avail.assert_not_called()


# ---------------------------------------------------------------------------
# get(): retry via exit node
# ---------------------------------------------------------------------------


def test_direct_403_retries_via_exit_node_proxy(exit_node_on):
    forbidden, ok = _resp(403), _resp(200)
    with patch(
        "paper_fetch_mcp.core.exit_node.requests.get", side_effect=[forbidden, ok]
    ) as g:
        r = exit_node.get("https://ieeexplore.ieee.org/document/1", timeout=5, headers={"A": "b"})
    assert r.response is ok
    assert r.route == "exit_node"
    assert r.exit_node == "tung-lab-pc"
    assert r.direct_error == "HTTP 403"
    assert g.call_count == 2
    first, second = g.call_args_list
    assert "proxies" not in first.kwargs
    assert second.kwargs["proxies"] == {
        "http": "socks5h://127.0.0.1:1056",
        "https": "socks5h://127.0.0.1:1056",
    }
    # the original request's own kwargs are preserved on the retry
    assert second.kwargs["headers"] == {"A": "b"}
    assert second.kwargs["timeout"] == 5


def test_direct_connection_error_retries_via_exit_node(exit_node_on):
    ok = _resp(200)
    with patch(
        "paper_fetch_mcp.core.exit_node.requests.get",
        side_effect=[requests.ConnectionError("refused"), ok],
    ) as g:
        r = exit_node.get("https://example.org/p", timeout=5)
    assert r.response is ok and r.route == "exit_node"
    assert r.direct_error.startswith("ConnectionError")
    assert g.call_args_list[1].kwargs["proxies"]["https"] == "socks5h://127.0.0.1:1056"


def test_exit_node_attempt_also_failing_raises_error_naming_both(exit_node_on):
    with patch(
        "paper_fetch_mcp.core.exit_node.requests.get",
        side_effect=[requests.Timeout("direct slow"), requests.ConnectionError("proxy down")],
    ):
        with pytest.raises(exit_node.ExitNodeFetchError) as ei:
            exit_node.get("https://example.org/p", timeout=5)
    msg = str(ei.value)
    assert "direct: Timeout" in msg
    assert "via exit node tung-lab-pc" in msg and "proxy down" in msg
    # still a requests exception, so `except requests.RequestException` catches it
    assert isinstance(ei.value, requests.RequestException)


def test_exit_node_403_is_returned_with_route_for_caller_to_raise(exit_node_on):
    """If the exit node ALSO says 403 the response is handed back (route
    exit_node) so the caller's raise_for_status produces the error -- the
    route layer does not swallow HTTP errors."""
    with patch(
        "paper_fetch_mcp.core.exit_node.requests.get", side_effect=[_resp(403), _resp(403)]
    ):
        r = exit_node.get("https://example.org/p", timeout=5)
    assert r.route == "exit_node" and r.response.status_code == 403
    assert r.direct_error == "HTTP 403"


# ---------------------------------------------------------------------------
# get(): exit node unavailable
# ---------------------------------------------------------------------------


def test_unavailable_exit_node_returns_direct_response_with_reason(exit_node_unavailable):
    forbidden = _resp(403)
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=forbidden) as g:
        r = exit_node.get("https://example.org/p", timeout=5)
    assert r.response is forbidden and r.route == "direct"
    assert r.direct_error == "HTTP 403"
    assert "not listening" in r.skipped_reason
    assert g.call_count == 1


def test_unavailable_exit_node_reraises_direct_exception(exit_node_unavailable):
    with patch(
        "paper_fetch_mcp.core.exit_node.requests.get",
        side_effect=requests.ConnectionError("refused"),
    ) as g:
        with pytest.raises(requests.ConnectionError):
            exit_node.get("https://example.org/p", timeout=5)
    assert g.call_count == 1


def test_mode_off_never_probes_or_retries(monkeypatch):
    """conftest sets EXIT_NODE_MODE=off; availability() must short-circuit
    without touching the network, and get() must not retry."""
    with patch("paper_fetch_mcp.core.exit_node._tcp_probe") as probe, \
         patch("paper_fetch_mcp.core.exit_node._sidecar_exit_node_state") as state:
        avail = exit_node.availability(force_refresh=True)
        assert avail == {
            "mode": "off",
            "exit_node": config.EXIT_NODE_NAME,
            "proxy_url": config.EXIT_NODE_PROXY_URL,
            "available": False,
            "reason": "EXIT_NODE_MODE=off",
        }
        probe.assert_not_called()
        state.assert_not_called()
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=_resp(403)) as g:
        r = exit_node.get("https://example.org/p", timeout=5)
    assert r.route == "direct" and r.skipped_reason == "EXIT_NODE_MODE=off"
    assert g.call_count == 1


# ---------------------------------------------------------------------------
# get(force_exit_node=True)
# ---------------------------------------------------------------------------


def test_force_exit_node_goes_straight_through_proxy(exit_node_on):
    ok = _resp(200)
    with patch("paper_fetch_mcp.core.exit_node.requests.get", return_value=ok) as g:
        r = exit_node.get("https://example.org/p", timeout=5, force_exit_node=True)
    assert r.route == "exit_node" and r.response is ok and r.direct_error is None
    assert g.call_count == 1
    assert g.call_args.kwargs["proxies"]["https"] == "socks5h://127.0.0.1:1056"


def test_force_exit_node_when_unavailable_raises(exit_node_unavailable):
    with patch("paper_fetch_mcp.core.exit_node.requests.get") as g:
        with pytest.raises(exit_node.ExitNodeUnavailable, match="not listening"):
            exit_node.get("https://example.org/p", timeout=5, force_exit_node=True)
    g.assert_not_called()


# ---------------------------------------------------------------------------
# availability(): probe logic and caching
# ---------------------------------------------------------------------------


def test_availability_proxy_port_closed(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    monkeypatch.setattr(config, "EXIT_NODE_PROXY_URL", "socks5h://127.0.0.1:1056")
    with patch("paper_fetch_mcp.core.exit_node._tcp_probe", return_value=False) as probe, \
         patch("paper_fetch_mcp.core.exit_node._sidecar_exit_node_state") as state:
        avail = exit_node.availability(force_refresh=True)
    probe.assert_called_once_with("127.0.0.1", 1056)
    state.assert_not_called()
    assert avail["available"] is False
    assert "127.0.0.1:1056 is not listening" in avail["reason"]
    assert "exit-node sidecar" in avail["reason"]


def test_availability_sidecar_without_engaged_exit_node(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    with patch("paper_fetch_mcp.core.exit_node._tcp_probe", return_value=True), \
         patch(
             "paper_fetch_mcp.core.exit_node._sidecar_exit_node_state",
             return_value=(False, "sidecar tailscaled is NeedsLogin, not Running"),
         ):
        avail = exit_node.availability(force_refresh=True)
    assert avail["available"] is False
    assert "NeedsLogin" in avail["reason"]


def test_availability_engaged(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    with patch("paper_fetch_mcp.core.exit_node._tcp_probe", return_value=True), \
         patch(
             "paper_fetch_mcp.core.exit_node._sidecar_exit_node_state",
             return_value=(True, "exit node engaged"),
         ):
        avail = exit_node.availability(force_refresh=True)
    assert avail["available"] is True and avail["exit_node"] == config.EXIT_NODE_NAME


def test_availability_unknown_sidecar_state_trusts_port_probe(monkeypatch):
    """No tailscale CLI / socket -> we cannot ask the sidecar; a listening
    proxy port is then taken at face value."""
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    with patch("paper_fetch_mcp.core.exit_node._tcp_probe", return_value=True), \
         patch(
             "paper_fetch_mcp.core.exit_node._sidecar_exit_node_state",
             return_value=(None, "sidecar state unknown (no tailscale CLI)"),
         ):
        avail = exit_node.availability(force_refresh=True)
    assert avail["available"] is True


def test_availability_is_cached_within_ttl(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_MODE", "auto")
    with patch("paper_fetch_mcp.core.exit_node._tcp_probe", return_value=False) as probe:
        exit_node.availability(force_refresh=True)
        exit_node.availability()
        exit_node.availability()
    assert probe.call_count == 1


# ---------------------------------------------------------------------------
# _sidecar_exit_node_state(): parsing `tailscale --socket <sidecar> status --json`
# ---------------------------------------------------------------------------


def _completed(stdout: str, rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def test_sidecar_state_running_with_online_exit_node(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_TAILSCALE_SOCKET", "/tmp/x.sock")
    payload = json.dumps(
        {"BackendState": "Running", "ExitNodeStatus": {"ID": "n1", "Online": True}}
    )
    with patch("paper_fetch_mcp.core.exit_node.shutil.which", return_value="/usr/bin/tailscale"), \
         patch("paper_fetch_mcp.core.exit_node.subprocess.run", return_value=_completed(payload)) as run:
        engaged, detail = exit_node._sidecar_exit_node_state()
    assert engaged is True
    assert run.call_args.args[0] == ["/usr/bin/tailscale", "--socket=/tmp/x.sock", "status", "--json"]


def test_sidecar_state_needs_login(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_TAILSCALE_SOCKET", "/tmp/x.sock")
    payload = json.dumps({"BackendState": "NeedsLogin", "ExitNodeStatus": None})
    with patch("paper_fetch_mcp.core.exit_node.shutil.which", return_value="/usr/bin/tailscale"), \
         patch("paper_fetch_mcp.core.exit_node.subprocess.run", return_value=_completed(payload)):
        engaged, detail = exit_node._sidecar_exit_node_state()
    assert engaged is False and "NeedsLogin" in detail


def test_sidecar_state_running_but_no_exit_node_set(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_TAILSCALE_SOCKET", "/tmp/x.sock")
    payload = json.dumps({"BackendState": "Running", "ExitNodeStatus": None})
    with patch("paper_fetch_mcp.core.exit_node.shutil.which", return_value="/usr/bin/tailscale"), \
         patch("paper_fetch_mcp.core.exit_node.subprocess.run", return_value=_completed(payload)):
        engaged, detail = exit_node._sidecar_exit_node_state()
    assert engaged is False and "no exit node set" in detail


def test_sidecar_state_exit_node_offline(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_TAILSCALE_SOCKET", "/tmp/x.sock")
    payload = json.dumps({"BackendState": "Running", "ExitNodeStatus": {"ID": "n1", "Online": False}})
    with patch("paper_fetch_mcp.core.exit_node.shutil.which", return_value="/usr/bin/tailscale"), \
         patch("paper_fetch_mcp.core.exit_node.subprocess.run", return_value=_completed(payload)):
        engaged, detail = exit_node._sidecar_exit_node_state()
    assert engaged is False and "offline" in detail


def test_sidecar_state_unknown_without_cli(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_TAILSCALE_SOCKET", "/tmp/x.sock")
    with patch("paper_fetch_mcp.core.exit_node.shutil.which", return_value=None), \
         patch("paper_fetch_mcp.core.exit_node.subprocess.run") as run:
        engaged, _ = exit_node._sidecar_exit_node_state()
    assert engaged is None
    run.assert_not_called()


def test_sidecar_state_unknown_on_cli_failure(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_TAILSCALE_SOCKET", "/tmp/x.sock")
    with patch("paper_fetch_mcp.core.exit_node.shutil.which", return_value="/usr/bin/tailscale"), \
         patch(
             "paper_fetch_mcp.core.exit_node.subprocess.run",
             side_effect=subprocess.TimeoutExpired(cmd="tailscale", timeout=5),
         ):
        engaged, detail = exit_node._sidecar_exit_node_state()
    assert engaged is None and "TimeoutExpired" in detail


# ---------------------------------------------------------------------------
# proxies() / _proxy_host_port()
# ---------------------------------------------------------------------------


def test_proxies_mapping_uses_configured_url(monkeypatch):
    monkeypatch.setattr(config, "EXIT_NODE_PROXY_URL", "socks5h://10.0.0.1:9999")
    assert exit_node.proxies() == {
        "http": "socks5h://10.0.0.1:9999",
        "https": "socks5h://10.0.0.1:9999",
    }


@pytest.mark.parametrize(
    "url,expected",
    [
        ("socks5h://127.0.0.1:1056", ("127.0.0.1", 1056)),
        ("http://localhost:1057", ("localhost", 1057)),
        ("127.0.0.1:1080", ("127.0.0.1", 1080)),
    ],
)
def test_proxy_host_port(url, expected):
    assert exit_node._proxy_host_port(url) == expected
