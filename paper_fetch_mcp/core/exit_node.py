"""Exit-node routing for online paper access.

When a paper is not in the local corpus, `get_paper` (via `core.online` and
`core.publisher`) fetches it from the network. Some publishers are only
reachable from the lab network (institutional IEEE/Springer/Elsevier
access), and some hosts sit behind networks that block or throttle those
sites outright. This module gives every such fetch a single, uniform
fallback: try the host's own network first, and if that attempt fails for
an *access* reason, retry once through a Tailscale exit node -- by default
`ubuntu-lab` (see `config.EXIT_NODE_*`).

How the exit node is reached
----------------------------
NOT by `tailscale set --exit-node=...` on the host. Two things rule that
out on this deployment, and both would make the toggle a footgun elsewhere:

* the host advertises an exit node itself, and Tailscale refuses to both
  advertise and use one ("Cannot advertise an exit node and use an exit
  node at the same time");
* its tailscaled runs with `--tun=userspace-networking` (the container has
  no NET_ADMIN), so a host-level exit node would not affect plain sockets
  anyway -- only traffic already going through tailscaled's SOCKS proxy.

Instead a second userspace tailscaled (the "exit-node sidecar", which this
repository does not ship) joins the tailnet as its own node,
uses `EXIT_NODE_NAME` as *its* exit node, and exposes a SOCKS5 proxy on
`EXIT_NODE_PROXY_URL`. Routing a request via the exit node is then just
`requests.get(..., proxies=proxies())` -- per request, thread-safe, and
with zero effect on anything else running on the host.

What counts as an access failure
--------------------------------
`looks_like_access_failure`: a connection-level error (refused, reset,
DNS, TLS, timeout) or an HTTP status that signals *this client* is not
allowed rather than that the resource is missing: 401, 403, 407, 429, 451.
A 404 or a 500 is NOT retried -- a different egress IP will not help.
Callers with their own notion of "access denied" (e.g. `core.publisher`'s
"200 OK but the page is a paywall stub" case) pass `force_exit_node=True`
to `get()` for a deliberate second attempt.

No side effects at import time (core/__init__.py rule): the availability
probe (a TCP connect to the proxy port plus an optional
`tailscale --socket <sidecar> status --json`) runs lazily, on first use,
and is cached for `_PROBE_TTL_S` seconds.
"""

from __future__ import annotations

import json
import logging
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

import requests

from paper_fetch_mcp import config

logger = logging.getLogger(__name__)

# HTTP statuses that mean "you (this network) may not have this", as
# opposed to "this does not exist" / "the server is broken".
ACCESS_FAILURE_STATUSES = frozenset({401, 403, 407, 429, 451})

_PROBE_TTL_S = 10.0
_TCP_PROBE_TIMEOUT_S = 0.75
_TAILSCALE_STATUS_TIMEOUT_S = 5.0

_probe_lock = threading.Lock()
_probe_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)


class ExitNodeUnavailable(RuntimeError):
    """Raised by `get(force_exit_node=True)` when the exit-node route is not
    usable (mode off, sidecar down, or its exit node not engaged)."""


class ExitNodeFetchError(requests.RequestException):
    """Both the direct attempt and the exit-node retry failed at the
    connection level. Subclasses `requests.RequestException` so existing
    `except requests.RequestException` handlers keep working; the message
    names both failures."""


@dataclass
class RouteResult:
    """A response plus the route that produced it.

    `route` is "direct" or "exit_node". `direct_error` is set when the
    direct attempt failed and the exit node was (or would have been) tried;
    `skipped_reason` is set when a direct access failure was NOT retried
    because the exit node was unavailable -- the caller still gets the
    direct response so it can `raise_for_status()` as usual.
    """

    response: requests.Response
    route: str = "direct"
    exit_node: Optional[str] = None
    direct_error: Optional[str] = None
    skipped_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def looks_like_access_failure(
    exc: Optional[BaseException] = None,
    response: Optional[requests.Response] = None,
) -> bool:
    """True if `exc`/`response` indicates the *network path* is the
    problem, not the resource. See module docstring."""
    if exc is not None:
        return isinstance(exc, (requests.ConnectionError, requests.Timeout))
    if response is not None:
        return response.status_code in ACCESS_FAILURE_STATUSES
    return False


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def _proxy_host_port(proxy_url: str) -> Tuple[str, int]:
    parts = urlsplit(proxy_url if "://" in proxy_url else f"socks5h://{proxy_url}")
    host = parts.hostname or "127.0.0.1"
    port = parts.port or 1080
    return host, port


def _tcp_probe(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=_TCP_PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def _sidecar_exit_node_state() -> Tuple[Optional[bool], str]:
    """Ask the sidecar tailscaled whether its exit node is engaged.

    Returns (engaged, detail): `engaged` is True/False when the sidecar
    answered, or None when it could not be asked (no `tailscale` CLI, no
    socket, timeout) -- in which case the caller falls back to trusting
    the TCP probe alone.
    """
    sock = config.EXIT_NODE_TAILSCALE_SOCKET
    cli = shutil.which("tailscale")
    if not sock or not cli:
        return None, "sidecar state unknown (no tailscale CLI or socket configured)"
    try:
        proc = subprocess.run(
            [cli, f"--socket={sock}", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_TAILSCALE_STATUS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"sidecar state unknown ({type(e).__name__}: {e})"
    if proc.returncode != 0:
        return None, f"sidecar state unknown ({(proc.stderr or proc.stdout).strip()[:200]})"
    try:
        status = json.loads(proc.stdout)
    except ValueError:
        return None, "sidecar state unknown (unparseable status JSON)"

    backend = status.get("BackendState")
    if backend != "Running":
        return False, f"sidecar tailscaled is {backend or 'unknown'}, not Running (log the sidecar in)"
    exit_status = status.get("ExitNodeStatus") or {}
    if not exit_status.get("ID"):
        return False, f"sidecar has no exit node set (want {config.EXIT_NODE_NAME})"
    if exit_status.get("Online") is False:
        return False, f"exit node {config.EXIT_NODE_NAME} is offline"
    return True, f"exit node engaged via sidecar (BackendState=Running)"


def _probe() -> Dict[str, Any]:
    mode = config.EXIT_NODE_MODE
    name = config.EXIT_NODE_NAME
    proxy_url = config.EXIT_NODE_PROXY_URL
    info: Dict[str, Any] = {"mode": mode, "exit_node": name, "proxy_url": proxy_url}

    if mode == "off":
        return {**info, "available": False, "reason": "EXIT_NODE_MODE=off"}
    if not name:
        return {**info, "available": False, "reason": "EXIT_NODE_NAME is empty"}

    host, port = _proxy_host_port(proxy_url)
    if not _tcp_probe(host, port):
        return {
            **info,
            "available": False,
            "reason": (
                f"exit-node proxy {host}:{port} is not listening "
                "(start the exit-node sidecar)"
            ),
        }

    engaged, detail = _sidecar_exit_node_state()
    if engaged is False:
        return {**info, "available": False, "reason": detail}
    return {**info, "available": True, "reason": detail}


def availability(force_refresh: bool = False) -> Dict[str, Any]:
    """Whether the exit-node route is usable right now, and why not if not.

    `{mode, exit_node, proxy_url, available: bool, reason}`. Cached for
    `_PROBE_TTL_S` seconds so a burst of fetches pays for one probe.
    """
    global _probe_cache
    now = time.monotonic()
    with _probe_lock:
        ts, cached = _probe_cache
        if cached is not None and not force_refresh and now - ts < _PROBE_TTL_S:
            return dict(cached)
    result = _probe()
    with _probe_lock:
        _probe_cache = (time.monotonic(), result)
    return dict(result)


def proxies() -> Dict[str, str]:
    """`requests`-style proxies mapping that routes a request via the exit
    node (both schemes through the sidecar's SOCKS5)."""
    url = config.EXIT_NODE_PROXY_URL
    return {"http": url, "https": url}


def status() -> Dict[str, Any]:
    """Operator-facing summary for `server_status`."""
    return availability()


# ---------------------------------------------------------------------------
# Routed GET
# ---------------------------------------------------------------------------


def get(
    url: str,
    *,
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    allow_redirects: bool = True,
    force_exit_node: bool = False,
    direct_only: bool = False,
) -> RouteResult:
    """GET `url`, retrying once via the exit node on an access failure.

    0. `direct_only=True`: GET on this host's own network only and return
       the response as-is -- no exit-node retry, whatever the outcome. For
       callers that have further own-network attempts to make (e.g. a
       headless browser) before falling back to the exit node.
    1. `force_exit_node=True`: go straight through the exit node (raises
       `ExitNodeUnavailable` if it cannot be used).
    2. Otherwise GET directly. A response whose status is NOT in
       `ACCESS_FAILURE_STATUSES` is returned as-is (including 404/5xx --
       the caller decides what to do with those).
    3. On a connection error / timeout / access status: if the exit node
       is available, GET again with `proxies()` and return that response
       (`route="exit_node"`, `direct_error` set). If it is not, return the
       direct response with `skipped_reason` set, or re-raise the direct
       exception if there was no response at all.
    4. If the exit-node attempt itself fails at the connection level,
       raise `ExitNodeFetchError` naming both failures.
    """
    kwargs: Dict[str, Any] = {
        "headers": headers,
        "params": params,
        "timeout": timeout,
        "allow_redirects": allow_redirects,
    }
    name = config.EXIT_NODE_NAME

    if direct_only:
        return RouteResult(response=requests.get(url, **kwargs))

    if force_exit_node:
        avail = availability()
        if not avail["available"]:
            raise ExitNodeUnavailable(avail["reason"])
        logger.info("fetching %s via exit node %s (forced)", url, name)
        resp = requests.get(url, proxies=proxies(), **kwargs)
        return RouteResult(response=resp, route="exit_node", exit_node=name)

    direct_resp: Optional[requests.Response] = None
    direct_exc: Optional[BaseException] = None
    try:
        direct_resp = requests.get(url, **kwargs)
        if not looks_like_access_failure(response=direct_resp):
            return RouteResult(response=direct_resp)
        direct_error = f"HTTP {direct_resp.status_code}"
    except (requests.ConnectionError, requests.Timeout) as e:
        direct_exc = e
        direct_error = f"{type(e).__name__}: {e}"

    avail = availability()
    if not avail["available"]:
        logger.info(
            "direct fetch of %s failed (%s); exit node not used: %s",
            url, direct_error, avail["reason"],
        )
        if direct_resp is not None:
            return RouteResult(
                response=direct_resp,
                direct_error=direct_error,
                skipped_reason=avail["reason"],
            )
        assert direct_exc is not None
        raise direct_exc

    logger.info(
        "direct fetch of %s failed (%s); retrying via exit node %s", url, direct_error, name
    )
    try:
        resp = requests.get(url, proxies=proxies(), **kwargs)
    except requests.RequestException as e:
        raise ExitNodeFetchError(
            f"direct: {direct_error}; via exit node {name}: {type(e).__name__}: {e}"
        ) from e
    return RouteResult(
        response=resp, route="exit_node", exit_node=name, direct_error=direct_error
    )
