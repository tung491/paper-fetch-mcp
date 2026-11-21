"""Publisher HTML fetch: resolve identifiers, fetch (requests-first), cache.

Moved here from ``paper_fetch_mcp/tools/publisher_fetch.py`` per the refactor
design spec §5 "Target architecture"
(docs/superpowers/specs/2026-09-13-agentic-rag-mcp-refactor-design.md).

``ensure_cache_table`` (the ``cached_html`` table DDL) was relocated here
from ``paper_fetch_mcp/db.py`` -- that table is publisher-cache state, not
paper metadata, and ``get_cache``/``upsert_cache`` below are its only real
readers. Per package convention (see ``core/__init__.py``), it is NOT called
at import time; the FastMCP lifespan is responsible for calling it.

Uses ``core.papers.get_db_connection``/``resolve_doi_to_id`` (not
``paper_fetch_mcp/db.py``) so importing this module never triggers
``db.py``'s module-level ``_ensure_cached_html_table()`` call -- that was
the last import-time DB side effect in the package (Wave 5b, task 2).

Fetch strategy (Phase 2, spec §6/§8 Phase 2, finding F9): ``fetch_via_dev_browser``
(Selenium/stealth Chrome) logged "Failed to start Chrome" in 5 of 8 production
calls -- Selenium needs a real browser on what is a headless box. ``orchestrate_fetch``
now tries a plain ``requests.get`` + the existing BeautifulSoup/markdownify
extraction (``fetch_via_requests`` / ``extract_paper_content``) FIRST, and
only falls back to Selenium when the ``PUBLISHER_FETCH_BROWSER=1`` environment
variable is set (checked at call time, not at import, so tests and ordinary
deployments never touch Chrome unless an operator opts in). Either path
returns extracted **markdown**, never raw HTML.

Exit-node route (2026-09-14, `core.exit_node`): every outbound GET here
(`_resolve_doi_url`, `fetch_via_requests_routed`) goes through
``exit_node.get``, which can retry via the Tailscale exit node (default
`ubuntu-lab`) when the host's own network is refused.

Own network first (2026-10-24): ``orchestrate_fetch`` exhausts this host's
own network before touching the exit node -- a plain requests GET
(``direct_only``), then a headless Chromium render
(``fetch_via_headless_browser``), which passes JS bot challenges such as
IEEE Xplore's AWS WAF (an empty HTTP 202) that no plain GET can. Only when
both fail -- an error, or a body below ``_MIN_EXTRACTED_CONTENT_CHARS``
(paywall/consent stub) -- is the page fetched once via the exit node, then
the Selenium / Semantic Scholar fallbacks. The result carries a ``route``
field ("direct" | "exit_node" | "cache") so a caller can tell which network
actually served the page.
"""

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import yaml

from .. import config
from . import exit_node
from . import images
from . import papers

log = logging.getLogger(__name__)

# Below this many extracted characters, a "successful" requests+extraction
# fetch is treated as having found nothing usable (e.g. a paywall/consent
# page with no article body) and the next stage of the fallback chain is
# tried instead of returning a near-empty result.
_MIN_EXTRACTED_CONTENT_CHARS = 200

# A cached `source: "fallback"` row (Semantic Scholar abstract only, written
# when every real fetch failed) is re-tried instead of served from cache while
# the exit node is available and the row has failed fewer than this many
# times -- otherwise a paper first looked up while the exit-node sidecar was
# down would be pinned to its abstract forever. `error_count` grows by one
# per fallback, so this bounds the retries per identifier.
_MAX_FALLBACK_RETRIES_VIA_EXIT_NODE = 3

SS_GRAPH_API = "https://api.semanticscholar.org/graph/v1/paper"
# Optional: Semantic Scholar's Graph API accepts unauthenticated requests at a
# lower rate limit, so an unset key is a valid state -- see _ss_headers().
SS_API_KEY = config.SEMANTIC_SCHOLAR_API_KEY


def _ss_headers() -> dict:
    """Auth header for Semantic Scholar Graph API calls.

    Never send an empty string as the x-api-key value: omit the header
    entirely when SEMANTIC_SCHOLAR_API_KEY is not configured. S2 still serves
    unauthenticated requests, just at a lower rate limit.
    """
    return {"x-api-key": SS_API_KEY} if SS_API_KEY else {}


# ---------------------------------------------------------------------------
# cached_html table (publisher-cache state; relocated from db.py)
# ---------------------------------------------------------------------------

def ensure_cache_table() -> None:
    """Create the ``cached_html`` table if it does not exist (idempotent).

    Deliberately NOT called at import time -- call this from the FastMCP
    lifespan (see core/__init__.py's "no side effects at import" rule).
    """
    try:
        with papers.get_db_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_html (
                    id           TEXT PRIMARY KEY,
                    url          TEXT NOT NULL,
                    source       TEXT NOT NULL,
                    html         TEXT NOT NULL,
                    fetched_at   TEXT NOT NULL,
                    error_count  INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()
            log.info("cached_html table ensured")
    except Exception as e:
        log.warning("cached_html table init failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# DOI -> publisher URL resolution
# ---------------------------------------------------------------------------

def _resolve_doi_url(doi: str) -> Tuple[str, str]:
    """
    Resolve a DOI to a publisher URL via https://doi.org redirect.

    Returns (url, source) where source is one of:
      IEEE | Springer | ScienceDirect | Other
    """
    try:
        routed = exit_node.get(
            f"https://doi.org/{doi}",
            headers={"User-Agent": "AgenticRAGServer/1.0"},
            allow_redirects=False,
            timeout=15,
        )
        resp = routed.response
        final_url = resp.headers.get("Location", resp.url)
    except requests.RequestException as e:
        raise ValueError(f"Failed to resolve DOI {doi}: {e}") from e

    source = _url_to_source(final_url)
    return final_url, source


def _url_to_source(url: str) -> str:
    """Classify a URL by its host into a source string."""
    url_lower = url.lower()
    if "ieeexplore.ieee.org" in url_lower:
        return "IEEE"
    if "link.springer.com" in url_lower:
        return "Springer"
    if "sciencedirect.com" in url_lower:
        return "ScienceDirect"
    if "arxiv.org" in url_lower:
        return "arXiv"
    return "Other"


# ---------------------------------------------------------------------------
# Semantic Scholar paperId -> DOI
# ---------------------------------------------------------------------------

def _resolve_semantic_scholar_paper_id(paper_id: str) -> Optional[str]:
    """
    Call the Semantic Scholar Graph API to get the DOI for a paperId.
    Returns the DOI string, or None if not found.
    """
    url = f"{SS_GRAPH_API}/{paper_id}"
    try:
        resp = requests.get(
            url,
            params={"fields": "externalIds"},
            headers=_ss_headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning("SS API returned %s for paperId %s", resp.status_code, paper_id)
            return None
        data = resp.json()
        doi = data.get("externalIds", {}).get("DOI")
        return doi if doi else None
    except Exception as e:
        log.warning("SS API call failed for paperId %s: %s", paper_id, e)
        return None


# ---------------------------------------------------------------------------
# Main identifier classifier
# ---------------------------------------------------------------------------

def classify_identifier(identifier: str) -> Tuple[str, str]:
    """
    Classify an identifier (DOI, paperId, or direct URL) into a publisher URL
    and source string.

    Resolution order:
      1. Direct URL (known publisher host)
      2. DOI → look up papers.db → publisher field OR doi.org redirect
      3. Semantic Scholar paperId → Graph API → DOI → url

    Returns (url, source).
    Raises ValueError on unresolvable input.
    """
    identifier = identifier.strip()

    # 1. Direct URL?
    if identifier.startswith("http://") or identifier.startswith("https://"):
        source = _url_to_source(identifier)
        return identifier, source

    # 2. DOI (starts with 10.)?
    if identifier.startswith("10."):
        # Try DB first for publisher info, then doi.org
        paper_data = papers.resolve_doi_to_id(identifier)
        if paper_data and paper_data.get("publisher"):
            pub = paper_data["publisher"].strip()
            # Still need URL from doi.org; use DB source
            final_url, _ = _resolve_doi_url(identifier)
            return final_url, pub
        # Fall back to doi.org redirect
        return _resolve_doi_url(identifier)

    # 3. Assume Semantic Scholar paperId
    doi = _resolve_semantic_scholar_paper_id(identifier)
    if not doi:
        raise ValueError(
            f"Could not resolve paperId '{identifier}' via Semantic Scholar. "
            "Please provide a DOI or direct publisher URL."
        )
    final_url, source = _resolve_doi_url(doi)
    return final_url, source


# ---------------------------------------------------------------------------
# requests-first fetch (spec §6/§8 Phase 2, finding F9) -- tried before ever
# touching Selenium/Chrome.
# ---------------------------------------------------------------------------

_REQUESTS_FETCH_TIMEOUT_S = 20
_REQUESTS_FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)


class RequestsFetchError(Exception):
    """Raised when a plain (no-JS) `requests.get` fetch of a publisher page
    fails outright (network error or non-2xx status). Distinct from
    BrowserError, which is Selenium-specific."""
    pass


def fetch_via_requests_routed(
    url: str, *, force_exit_node: bool = False, direct_only: bool = False
) -> Tuple[str, str]:
    """
    Fetch `url` with a plain `requests.get` -- no browser, no JS execution,
    no bot-detection stealth. Tried FIRST, before Selenium: Selenium/Chrome
    needs a real display and failed with "Failed to start Chrome" in 5 of 8
    production `fetch_publisher_html` calls on this headless box (finding
    F9), while many publisher abstract/landing pages render their core text
    server-side and need no JS at all.

    The GET goes through ``exit_node.get``: the host's own network first,
    then one retry via the Tailscale exit node when the direct attempt is
    an access failure (see `core.exit_node`). ``force_exit_node=True``
    skips the direct attempt -- `orchestrate_fetch` uses it when a direct
    200 turned out to be a paywall stub. ``direct_only=True`` never touches
    the exit node -- `orchestrate_fetch` uses it for its first attempt, so
    every own-network option (incl. the headless browser) is tried before
    the exit node.

    Returns ``(html, route)`` where route is "direct" or "exit_node".
    Raises RequestsFetchError on any network failure or non-2xx response
    (the message names both routes when both were tried). Callers MUST run
    the HTML through `extract_paper_content()` before returning it to a
    client; this function never itself returns anything client-facing.
    """
    headers = {
        "User-Agent": _REQUESTS_FETCH_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml",
    }
    try:
        routed = exit_node.get(
            url,
            headers=headers,
            timeout=_REQUESTS_FETCH_TIMEOUT_S,
            allow_redirects=True,
            force_exit_node=force_exit_node,
            direct_only=direct_only,
        )
    except exit_node.ExitNodeUnavailable as e:
        raise RequestsFetchError(f"exit-node fetch of {url} unavailable: {e}") from e
    except requests.RequestException as e:
        raise RequestsFetchError(f"requests fetch of {url} failed: {e}") from e

    try:
        routed.response.raise_for_status()
    except requests.RequestException as e:
        where = f"via exit node {routed.exit_node}" if routed.route == "exit_node" else "direct"
        detail = f"{where}: {e}"
        if routed.direct_error and routed.route == "exit_node":
            detail = f"direct: {routed.direct_error}; {detail}"
        elif routed.skipped_reason:
            detail = f"{detail} (exit node not used: {routed.skipped_reason})"
        raise RequestsFetchError(f"requests fetch of {url} failed: {detail}") from e
    return routed.response.text, routed.route


def fetch_via_requests(url: str) -> str:
    """Compatibility wrapper over `fetch_via_requests_routed` returning just
    the HTML (the route is dropped). Same exceptions."""
    html, _route = fetch_via_requests_routed(url)
    return html


# ---------------------------------------------------------------------------
# Headless browser on this host's own network -- tried after a plain
# requests fetch and BEFORE the exit node. Some publishers answer every
# non-browser client with a JS bot challenge (IEEE Xplore: AWS WAF, an empty
# HTTP 202 with `x-amzn-waf-action: challenge`), which says nothing about
# whether this network has access; a real browser passes it, and the
# publisher still sees this host's (institutional) IP.
# ---------------------------------------------------------------------------

_HEADLESS_BROWSER_CANDIDATES = (
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
)
# The whole browser stage -- launch, challenge, render, figures. Keeps
# doi.org (15 s) + requests (20 s) + this + exit node (20 s) inside the 90 s
# tool timeout of get_paper / fetch_publisher_html. An IEEE document page
# renders in ~9 s.
_HEADLESS_BROWSER_TIMEOUT_S = 30
_DEVTOOLS_PORT_WAIT_S = 10
# The page counts as rendered once it is complete, not a bot challenge, and
# its DOM size has not changed for this many polls in a row.
_SETTLE_POLL_S = 0.5
_SETTLE_STABLE_POLLS = 3
_MAX_FIGURES = 60

_PAGE_STATE_JS = """JSON.stringify([
  document.readyState,
  document.body ? document.body.innerHTML.length : 0,
  !!(window.AwsWafIntegration || document.getElementById('challenge-container')
     || document.querySelector('#challenge-form, [id^="cf-chl"]')
     || document.title === 'Client Challenge')
])"""

# Runs inside the page, so figures / the PDF are requested with the cookies
# of the session that passed the challenge (a plain GET of IEEE's
# /mediastore/ or /stampPDF/ is refused). Same-origin only in practice:
# cross-origin figure hosts without CORS fail here and are downloaded plainly
# at save time instead. Only responses whose type starts with the given
# prefix are kept -- a no-access PDF request comes back as an HTML page.
_FETCH_IN_PAGE_JS = """(async (urls, typePrefix) => {
  const out = {};
  await Promise.all(urls.map(async (u) => {
    try {
      const r = await fetch(u, {credentials: 'include'});
      const type = r.headers.get('content-type') || '';
      if (!r.ok || !type.startsWith(typePrefix)) return;
      const buf = new Uint8Array(await r.arrayBuffer());
      let bin = '';
      for (let i = 0; i < buf.length; i += 0x8000) {
        bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
      }
      out[u] = [type, btoa(bin)];
    } catch (e) {}
  }));
  return out;
})(%s, %s)"""


def _headless_browser_bin() -> Optional[str]:
    """The Chromium/Chrome binary for `fetch_via_headless_browser`, or None
    when it is disabled (``PUBLISHER_HEADLESS_BROWSER=0``) or not installed.
    ``PUBLISHER_HEADLESS_BROWSER_BIN`` overrides the PATH lookup. Read at
    call time, like ``PUBLISHER_FETCH_BROWSER``."""
    if os.environ.get("PUBLISHER_HEADLESS_BROWSER", "1").strip() == "0":
        return None
    configured = os.environ.get("PUBLISHER_HEADLESS_BROWSER_BIN", "").strip()
    if configured:
        return shutil.which(configured)
    for name in _HEADLESS_BROWSER_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


class _DevToolsPage:
    """Just enough of the Chrome DevTools Protocol for one page target:
    send a command, wait for its reply (events are skipped)."""

    def __init__(self, ws_url: str, deadline: float):
        import websocket

        self._deadline = deadline
        self._next_id = 0
        try:
            self._ws = websocket.create_connection(
                ws_url, timeout=self._remaining(), suppress_origin=True
            )
        except Exception as e:  # noqa: BLE001 -- any connect failure is a BrowserError
            raise BrowserError(f"could not connect to the browser: {e}") from e

    def _remaining(self) -> float:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise BrowserError(f"headless browser timed out after {_HEADLESS_BROWSER_TIMEOUT_S}s")
        return left

    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        self._next_id += 1
        msg_id = self._next_id
        try:
            self._ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
            while True:
                self._ws.settimeout(self._remaining())
                msg = json.loads(self._ws.recv())
                if msg.get("id") == msg_id:
                    break
        except BrowserError:
            raise
        except Exception as e:  # noqa: BLE001 -- socket errors/timeouts
            raise BrowserError(f"{method} failed: {type(e).__name__}: {e}") from e
        if "error" in msg:
            raise BrowserError(f"{method} failed: {msg['error'].get('message')}")
        return msg.get("result") or {}

    def evaluate(self, expression: str, await_promise: bool = False) -> Any:
        result = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            awaitPromise=await_promise,
        )
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"].get("text") or "exception"
            raise BrowserError(f"page script failed: {detail}")
        return (result.get("result") or {}).get("value")

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


def _launch_headless_browser(binary: str, profile: str) -> Tuple[subprocess.Popen, str]:
    """Start Chromium with DevTools on a free local port; return the process
    and the WebSocket URL of its page."""
    proc = subprocess.Popen(
        [
            binary,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            f"--user-data-dir={profile}",
            f"--user-agent={_REQUESTS_FETCH_USER_AGENT}",
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=0",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    port_file = os.path.join(profile, "DevToolsActivePort")
    give_up = time.monotonic() + _DEVTOOLS_PORT_WAIT_S
    port = ""
    while time.monotonic() < give_up and proc.poll() is None:
        try:
            with open(port_file) as f:
                port = f.readline().strip()
        except OSError:
            pass
        if port:
            break
        time.sleep(0.1)
    if not port:
        proc.kill()
        raise BrowserError(
            f"headless browser exited {proc.returncode}" if proc.returncode is not None
            else "headless browser did not open its DevTools port"
        )
    # Never through an environment proxy: this is a loopback connection.
    session = requests.Session()
    session.trust_env = False
    try:
        targets = session.get(f"http://127.0.0.1:{port}/json/list", timeout=5).json()
        ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")
    except Exception as e:  # noqa: BLE001
        proc.kill()
        raise BrowserError(f"could not find the browser's page target: {e}") from e
    return proc, ws_url


def _wait_until_rendered(page: _DevToolsPage) -> None:
    last_len, stable = -1, 0
    while True:
        try:
            ready, length, challenged = json.loads(page.evaluate(_PAGE_STATE_JS))
        except BrowserError as e:
            if "timed out" in str(e):
                raise
            # "Execution context was destroyed": the challenge reloaded the page.
            ready, length, challenged = "loading", 0, True
        if ready == "complete" and not challenged and length > 0:
            stable = stable + 1 if length == last_len else 1
            if stable >= _SETTLE_STABLE_POLLS:
                return
        else:
            stable = 0
        last_len = length
        time.sleep(_SETTLE_POLL_S)


def fetch_via_headless_browser(
    url: str,
    figure_urls: Optional[Callable[[str], List[str]]] = None,
    pdf_url: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Render `url` in a throwaway headless Chromium on this host's own
    network, driven over the DevTools protocol, and return the rendered DOM.

    Waits until any bot challenge has reloaded into the real page and the
    DOM has stopped changing. If `figure_urls` is given it is called with
    that HTML; the image URLs it returns are then fetched from INSIDE the
    page -- i.e. with the session that passed the challenge -- and stored
    in the asset cache (`core.images`) for a later `save_path`. Likewise
    `pdf_url`: if it names a PDF for that HTML (a page without its full
    text, e.g. IEEE Early Access), the PDF is fetched in-page and cached.

    One browser per call, with a fresh profile dir, so concurrent calls
    never share state. A normal Chrome user agent is sent: the default
    "HeadlessChrome" one is served the challenge page instead of the
    article.

    Raises BrowserError if no browser is available, it fails, or the whole
    stage exceeds _HEADLESS_BROWSER_TIMEOUT_S. Failing to fetch figures or
    the PDF is not an error.
    """
    binary = _headless_browser_bin()
    if not binary:
        raise BrowserError("no headless browser (chromium/google-chrome) available")
    deadline = time.monotonic() + _HEADLESS_BROWSER_TIMEOUT_S
    with tempfile.TemporaryDirectory(
        prefix="publisher-headless-", ignore_cleanup_errors=True
    ) as profile:
        try:
            proc, ws_url = _launch_headless_browser(binary, profile)
        except OSError as e:
            raise BrowserError(f"headless browser failed to start: {e}") from e
        page = None
        try:
            page = _DevToolsPage(ws_url, deadline)
            page.call("Page.navigate", url=url)
            _wait_until_rendered(page)
            html = page.evaluate("document.documentElement.outerHTML") or ""
            if figure_urls and html:
                _fetch_in_page(page, figure_urls(html)[:_MAX_FIGURES], "image/", "figures")
            pdf = pdf_url(html) if pdf_url and html else None
            if pdf:
                _fetch_in_page(page, [pdf], "application/pdf", "PDF")
            return html
        finally:
            if page is not None:
                page.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _fetch_in_page(page: _DevToolsPage, urls: List[str], type_prefix: str, what: str) -> None:
    wanted = [u for u in urls if images.cached(u) is None]
    if not wanted:
        return
    try:
        got = page.evaluate(
            _FETCH_IN_PAGE_JS % (json.dumps(wanted), json.dumps(type_prefix)),
            await_promise=True,
        ) or {}
    except BrowserError as e:
        log.info("%s fetch in the headless browser failed: %s", what, e)
        return
    for asset_url, (mime, b64) in got.items():
        images.store(asset_url, base64.b64decode(b64), mime)
    log.info("cached %d/%d %s from the headless browser", len(got), len(wanted), what)


# ---------------------------------------------------------------------------
# Pages without their full text: fall back to the publisher's PDF
# ---------------------------------------------------------------------------

_IEEE_ARNUMBER_RE = re.compile(r'"pdfUrl":"[^"]*arnumber=(\d+)')
_CITATION_PDF_URL_RE = re.compile(r'<meta name="citation_pdf_url" content="([^"]+)"')


# Interstitials a publisher serves a non-browser client instead of the page:
# Springer's "Client Challenge" (assets under /_fs-ch-<token>/, a 3 KB page
# whose only text is a ~200-char "A required part of this site couldn't
# load" notice), AWS WAF (IEEE), Cloudflare. Rejected by `orchestrate_fetch`
# whatever their length -- Springer's clears _MIN_EXTRACTED_CONTENT_CHARS.
_BOT_CHALLENGE_RE = re.compile(
    r"<title>\s*(?:Client Challenge|Just a moment\.\.\.)\s*</title>"
    r"|/_fs-ch-[\w-]+/|AwsWafIntegration|id=\"challenge-(?:container|form)\"|id=\"cf-chl",
    re.IGNORECASE,
)


def is_bot_challenge(html: str) -> bool:
    """True when ``html`` is a JS bot-challenge interstitial, not the page."""
    return bool(html) and _BOT_CHALLENGE_RE.search(html) is not None


def lacks_full_text(html: str, source: str) -> bool:
    """True when a rendered publisher page has the abstract but not the
    article text. IEEE renders the full text into ``#article`` only when
    this network has access AND an HTML version exists -- Early Access
    papers (``"isEarlyAccess":true``) and scanned legacy papers are
    PDF-only. Only a real rendered document page counts (it carries
    ``xplGlobal.document.metadata``); a bot-challenge or error page is just
    "too little content"."""
    if source == "IEEE":
        return "xplGlobal.document.metadata" in html and 'id="article"' not in html
    if source == "arXiv":
        # The abstract page (/abs/<id>), not the LaTeXML full text.
        return 'name="citation_arxiv_id"' in html and "ltx_document" not in html
    return False


def pdf_only_url(html: str, source: str) -> Optional[str]:
    """The URL that serves the full-text PDF for a page that `lacks_full_text`,
    or None. IEEE: ``/stampPDF/getPDF.jsp?tp=&arnumber=N`` is the PDF itself
    (``/stamp/stamp.jsp``, the document's "PDF" link, is an HTML frame
    around it); it is only served to a session with access."""
    if not lacks_full_text(html, source):
        return None
    if source == "IEEE":
        m = _IEEE_ARNUMBER_RE.search(html)
        if m:
            return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={m.group(1)}&ref="
    if source == "arXiv":
        m = _CITATION_PDF_URL_RE.search(html)
        if m:
            return m.group(1)
    return None


_ARXIV_FETCH_TIMEOUT_S = 20
_ARXIV_PDF_TIMEOUT_S = 40


def fetch_arxiv_full_text(url: str) -> Tuple[Optional[Dict[str, str]], List[str]]:
    """arXiv's full text for an arXiv URL: the LaTeXML HTML version
    (``/html/<id>`` -- formulas, figures and references intact) if there is
    one, else the PDF's text under the abstract page (title, authors,
    abstract), with the PDF cached for `save_path`.

    The abstract page alone (``/abs/<id>``, what an arXiv id resolves to)
    has no paper text, which is why this goes elsewhere. Plain GETs, own
    network first (`exit_node.get`); arXiv has no bot challenge.

    Returns ``({content, url, route}, reasons)`` or ``(None, reasons)``.
    """
    arxiv_id = papers.extract_arxiv_id(url)
    if not arxiv_id:
        return None, [f"no arXiv id in {url}"]
    reasons: List[str] = []
    headers = {"User-Agent": _REQUESTS_FETCH_USER_AGENT}

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        routed = exit_node.get(html_url, headers=headers, timeout=_ARXIV_FETCH_TIMEOUT_S)
        resp = routed.response
        if resp.status_code == 200 and "ltx_document" in resp.text:
            content = extract_paper_content(resp.text, "arXiv", resp.url)
            if len(content.strip()) >= _MIN_EXTRACTED_CONTENT_CHARS:
                return {"content": content, "url": resp.url, "route": routed.route}, reasons
            reasons.append("arXiv HTML full text had too little extractable content")
        else:
            reasons.append(f"arXiv has no HTML full text (HTTP {resp.status_code})")
    except requests.RequestException as e:
        reasons.append(f"arXiv HTML full text: {e}")

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    pdf = images.cached(pdf_url) or images.download(
        pdf_url, kinds=("application/pdf",), timeout=_ARXIV_PDF_TIMEOUT_S
    )
    if pdf is None:
        reasons.append(f"arXiv PDF {pdf_url} could not be downloaded")
        return None, reasons
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    page_md = ""
    try:
        resp = exit_node.get(abs_url, headers=headers, timeout=_ARXIV_FETCH_TIMEOUT_S).response
        if resp.status_code == 200:
            page_md = extract_paper_content(resp.text, "arXiv", resp.url)
    except requests.RequestException as e:
        log.info("arXiv abstract page %s: %s", abs_url, e)
    content = with_pdf_full_text(page_md, pdf_url, pdf)
    return {"content": content, "url": abs_url, "route": "direct"}, reasons


def pdf_to_text(pdf_path: "os.PathLike[str]") -> str:
    """Plain text of a PDF: ``pdftotext`` (poppler -- fast, keeps the
    two-column reading order, rejoins hyphenated words) when installed,
    else pypdf. Formulas come out as flattened text and figures not at all;
    the PDF itself is kept for those."""
    binary = shutil.which("pdftotext")
    if binary:
        try:
            proc = subprocess.run(
                [binary, "-enc", "UTF-8", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.replace("\f", "\n\n").strip()
        except (OSError, subprocess.SubprocessError) as e:
            log.info("pdftotext failed on %s: %s", pdf_path, e)
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def with_pdf_full_text(page_md: str, pdf_link: str, pdf_path: "os.PathLike[str]") -> str:
    """The page's own Markdown (title, abstract, metadata) followed by the
    PDF's text. The PDF link stays in the content, so `save_path` copies the
    cached PDF next to the saved file (`core.images.localize`)."""
    try:
        text = pdf_to_text(pdf_path)
    except Exception as e:  # noqa: BLE001 -- a broken PDF still leaves the link
        log.warning("could not extract text from %s: %s", pdf_path, e)
        text = ""
    parts = [
        page_md.strip(),
        "## Full text (from the PDF)",
        f"[Full-text PDF]({pdf_link})",
        "*Extracted as plain text: formulas are flattened and figures are only in the PDF.*",
        text,
    ]
    return "\n\n".join(p for p in parts if p)


def _publisher_fetch_browser_enabled() -> bool:
    """Whether Selenium/Chrome fallback is allowed. Checked at CALL time (not
    import time) so tests and ordinary deployments never touch a real
    browser unless an operator explicitly opts in via the environment."""
    return os.environ.get("PUBLISHER_FETCH_BROWSER") == "1"


# ---------------------------------------------------------------------------
# dev-browser automation -- fallback only, gated by PUBLISHER_FETCH_BROWSER=1
# ---------------------------------------------------------------------------

BROWSER_TIMEOUT_SEC = 30

# Module-level WebDriver instance (lazy-initialized, reused across calls)
_driver = None


class BrowserError(Exception):
    """Raised when the browser fails to fetch a page."""
    pass


def _get_stealth_driver() -> "webdriver.Chrome":
    """
    Get or create a reusable stealth Chrome WebDriver session.
    Uses selenium-stealth to avoid bot detection.
    """
    global _driver
    if _driver is not None:
        try:
            _driver.current_url  # smoke-check the session is alive
            return _driver
        except Exception:
            try:
                _driver.quit()
            except Exception:
                pass
            _driver = None

    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium_stealth import stealth
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    _driver = webdriver.Chrome(service=service, options=options)

    stealth(
        _driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Linux x86_64",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    return _driver


def fetch_via_dev_browser(url: str) -> str:
    """
    Navigate to `url` in a stealth Chrome session and return the rendered page HTML.

    The browser session is reused across calls for efficiency. Each navigation
    overwrites the current page content.

    Raises BrowserError if Chrome or chromedriver is unavailable, or if the
    page fails to load within BROWSER_TIMEOUT_SEC.
    """
    import time
    from selenium.common.exceptions import WebDriverException

    try:
        driver = _get_stealth_driver()
    except Exception as e:
        raise BrowserError(
            f"Failed to start Chrome/selenium-stealth: {e}. "
            "Ensure Chrome and webdriver-manager are installed: "
            "pip install selenium selenium-stealth webdriver-manager"
        ) from e

    try:
        driver.get(url)
        # Wait for JS-rendered content to settle
        time.sleep(3)
        return driver.page_source
    except WebDriverException as e:
        raise BrowserError(f"Failed to load {url}: {e}") from e
    except Exception as e:
        raise BrowserError(f"Unexpected error fetching {url}: {e}") from e


# Article-body selectors per publisher. "parts" are concatenated in order
# (title, abstract + metadata, full text) when at least one matches; "body"
# is a first-match list, as before.
_ARTICLE_PARTS = {
    # IEEE Xplore's rendered document page: the title, the desktop abstract
    # block (abstract, "Published in", date, DOI) and #article (full text,
    # present only when this network has access). Picking these instead of
    # the largest <div> keeps the site header/footer/account menus out.
    "IEEE": ["h1.document-title", ".abstract-desktop-div", "#article"],
    # arXiv: the LaTeXML full-text page (/html/<id>) is one <article>; the
    # abstract page (/abs/<id>) has title, authors, dateline and abstract.
    "arXiv": ["article.ltx_document", "h1.title", ".authors", ".dateline", "blockquote.abstract"],
}
_ARTICLE_BODY = {
    "IEEE": [
        "#abstract", ".abstract", "section[itemprop=description]",
        "div.document-abstract", ".Abstracts",
    ],
    "Springer": [
        # .c-article-body = actual paper text (main content div)
        # .main-content = sectioned content (intro/body/refs)
        # <article> comes last — on Springer it wraps the masthead,
        #   not the article body itself
        ".c-article-body", ".main-content",
        "article",
    ],
    "ScienceDirect": [
        # .Body = main body text; <article> = full article with refs
        ".Body.u-font-serif", "#body",
        "article",
    ],
}

# MathJax v2 keeps each formula's TeX source in a <script type="math/tex">
# (or "math/tex; mode=display") next to the rendered glyph spans below.
_MATHJAX_RENDER_SELECTOR = (
    ".MathJax_Preview, .MathJax, .MathJax_Display, .MathJax_SVG, "
    ".MathJax_SVG_Display, .MathJax_CHTML, .MJX_Assistive_MathML, mjx-container"
)
_ICON_MAX_PX = 32


def _replace_math(soup, tokens: dict) -> None:
    """Swap every formula for a plain-text token (restored as ``$...$`` /
    ``$$...$$`` after markdownify, which would otherwise escape the TeX's
    ``_`` and ``*``) and drop MathJax's rendered glyph spans."""
    from bs4 import NavigableString

    def token_for(tex: str, display: bool) -> str:
        key = f"MATHTOKEN{len(tokens):05d}X"
        tex = tex.strip()
        tokens[key] = f"\n\n$$\n{tex}\n$$\n\n" if display else f"${tex}$"
        return key

    for script in soup.find_all("script", attrs={"type": lambda t: bool(t) and t.startswith("math/tex")}):
        display = "mode=display" in script["type"]
        script.replace_with(NavigableString(token_for(script.get_text(), display)))
    for tag in soup.select(_MATHJAX_RENDER_SELECTOR):
        if not tag.decomposed:
            tag.decompose()
    # LaTeXML (arXiv HTML): a numbered equation is a table row of MathML
    # cells plus an "(n)" tag cell -- one display formula per row, numbered
    # with \tag, instead of a Markdown table.
    for table in soup.select("table.ltx_equation, table.ltx_equationgroup"):
        rows = []
        for tr in table.find_all("tr"):
            tex = " ".join(_mathml_tex(m) for m in tr.find_all("math")).strip()
            if not tex:
                continue
            tag = tr.select_one(".ltx_tag_equation")
            if tag:
                tex += " \\tag{%s}" % tag.get_text(strip=True).strip("()")
            rows.append(token_for(tex, True))
        p = soup.new_tag("p")
        p.string = "".join(rows)
        table.replace_with(p)
    # MathML: the TeX is in alttext (or a TeX annotation).
    for m in soup.find_all("math"):
        tex = _mathml_tex(m)
        if tex:
            m.replace_with(NavigableString(token_for(tex, m.get("display") == "block")))
    # JATS-style <tex-math> that MathJax never processed: the TeX is its text.
    for tm in soup.find_all("tex-math"):
        text = tm.get_text()
        if text.strip() and "MATHTOKEN" not in text:
            display = tm.find_parent("disp-formula") is not None
            tm.string = token_for(text.strip().strip("$").strip(), display)


def _mathml_tex(math) -> str:
    tex = math.get("alttext") or ""
    if not tex:
        ann = math.find("annotation", attrs={"encoding": "application/x-tex"})
        tex = ann.get_text() if ann else ""
    return tex.strip()


def _tidy_publisher_markup(article, source: str) -> None:
    if source == "IEEE":
        # Each <disp-formula> also carries a hidden "View Source" copy of its
        # TeX -- the formula itself is already a token.
        for tag in article.select("disp-formula span.formula"):
            tag.decompose()
        # "<b>Fig. 3. </b><fig><p>caption</p></fig>": keep label and caption
        # on one line.
        for p in article.select(".figcaption fig p"):
            p.unwrap()


def _is_icon(img, src: str) -> bool:
    if "/assets/" in src or src.lower().endswith(".svg") and "logo" in src.lower():
        return True
    for dim in ("width", "height"):
        try:
            if int(str(img.get(dim, "")).rstrip("px")) <= _ICON_MAX_PX:
                return True
        except ValueError:
            pass
    return False


def _clean_images(article, base_url: str) -> None:
    """Keep article figures as ``<img src=ABSOLUTE alt=SHORT>``: prefer the
    full-size image a thumbnail links to, resolve lazy-load attributes, and
    drop icons / placeholders."""
    import re
    from urllib.parse import urljoin

    for img in article.find_all("img"):
        src = img.get("data-src") or img.get("data-lazy-src") or img.get("src") or ""
        if (not src or src.startswith("data:")) and img.get("srcset"):
            src = img["srcset"].split(",")[0].strip().split(" ")[0]
        link = img.find_parent("a")
        href = (link.get("href") or "") if link else ""
        if re.search(r"\.(gif|png|jpe?g|webp|svg|tiff?)(\?|$)", href, re.I):
            src = href
        if not src or src.startswith("data:") or _is_icon(img, src):
            img.decompose()
            continue
        alt = re.sub(r"\s+", " ", img.get("alt") or "").strip()
        if alt in ("", "Refer to caption"):  # LaTeXML's placeholder
            label = img.find_parent("figure")
            label = label.select_one("figcaption .ltx_tag") if label else None
            alt = label.get_text(strip=True).rstrip(":") if label else alt
        # IEEE: "Fig. 3. - <full caption>" -- the caption follows the image
        # anyway, so the alt text keeps just the label.
        m = re.match(r"((?:Fig\.|Figure|Table|Algorithm)\s*[\w.]+?)\.?\s+-\s", alt)
        alt = (m.group(1) if m else alt)[:120].replace("[", "(").replace("]", ")")
        img.attrs = {"src": urljoin(base_url, src) if base_url else src, "alt": alt}


def _clean_links(article, base_url: str) -> None:
    """Unwrap links that lead nowhere outside the page -- ``javascript:``
    handlers (IEEE's citation anchors), in-page ``#`` anchors, and links
    around images -- so "[12]" stays plain text; make the rest absolute."""
    from urllib.parse import urljoin

    for a in article.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "#")) or a.find("img"):
            a.unwrap()
        elif base_url:
            a["href"] = urljoin(base_url, href)


def extract_paper_content(html: str, source: str, base_url: str = "") -> str:
    """
    Extract clean paper content from publisher HTML and convert to Markdown.

    Uses BeautifulSoup to strip navigation, footers, cookie banners, and ads,
    then converts the remaining article body to Markdown via markdownify.
    Formulas become ``$TeX$`` / ``$$TeX$$``, figures ``![label](absolute
    URL)``, and citation / in-page links plain text.

    Args:
        html: Raw HTML from the browser.
        source: Publisher source string (IEEE, Springer, ScienceDirect, Other).
        base_url: The page's URL, to make image and link URLs absolute.

    Returns:
        Markdown-formatted paper content.
    """
    import re
    from bs4 import BeautifulSoup
    import markdownify

    soup = BeautifulSoup(html, "html.parser")

    # 0. Formulas to tokens BEFORE scripts go: the TeX lives in <script>s.
    math_tokens: dict = {}
    _replace_math(soup, math_tokens)

    # 1. Remove top-level noise elements (scripts/styles/etc.) from full page
    #    but DO NOT decompose layout wrappers that contain article content
    for tag in soup.find_all(
        ["script", "style", "noscript", "iframe", "form", "button", "input"]
    ):
        tag.decompose()

    # 2. Source-specific article body selectors — extract BEFORE noise removal
    #    so we don't accidentally remove the parent of article content
    article = None
    parts = []
    for sel in _ARTICLE_PARTS.get(source, []):
        found = soup.select_one(sel)
        if found and not any(anc is p for anc in found.parents for p in parts):
            parts.append(found)
    if parts:
        article = soup.new_tag("div")
        for part in parts:
            article.append(part.extract())
    for sel in [] if article else _ARTICLE_BODY.get(source, []):
        article = soup.select_one(sel)
        if article:
            break

    # Fallback: largest <section> or <div> with significant text
    if article is None:
        candidates = soup.find_all(["section", "div"])
        article = max(
            candidates,
            key=lambda t: len(t.get_text(strip=True)),
            default=soup.body if soup.body else soup,
        )

    # 3. Remove noise from inside the article element only
    #    (skip the article element itself — decomposing its parents removes content)
    noise_selectors = [
        # Structural noise tags -- whole names only: a substring match on
        # "form" also deleted every <inline-formula>/<disp-formula>.
        {"name": re.compile(r"^(?:script|style|noscript|iframe|svg|form|button|input|nav|footer|header)$")},
        # Cookie/consent banners
        {"class": re.compile(r"cookie|banner|consent|gdpr|ccpa|onetrust", re.I)},
        {"id": re.compile(r"cookie|banner|consent|gdpr|ccpa|onetrust", re.I)},
        # Layout noise (but not sd-flex-container which wraps article)
        {"class": re.compile(r"popup|modal|overlay|consent", re.I)},
        # Social/share noise
        {"class": re.compile(r"social|share|sidebar|related|suggested", re.I)},
    ]

    # An <svg> is noise, unless it carries an image (LaTeXML wraps some
    # figures in an annotation <svg>): keep the image, drop the drawing.
    for svg in article.find_all("svg"):
        imgs = svg.find_all("img")
        if imgs:
            svg.replace_with(*[img.extract() for img in imgs])

    for selector in noise_selectors:
        for tag in article.find_all(**selector):
            # Safety: never remove the article itself
            if tag is article or tag.decomposed:
                continue
            tag.decompose()

    _tidy_publisher_markup(article, source)
    _clean_images(article, base_url)
    _clean_links(article, base_url)

    # 4. Remove empty tags (an <img> has no text but is content)
    for tag in article.find_all():
        if tag is article or tag.decomposed:
            continue
        if tag.name == "img" or tag.find("img"):
            continue
        if not tag.get_text(strip=True):
            tag.decompose()

    # 5. Convert to markdown
    md = markdownify.markdownify(
        str(article),
        heading_style="ATX",
        bullets="-",
        links_text=True,
    )

    # Clean up excessive newlines
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    if not md or len(md) < 50:
        # Fallback: try the largest element with text
        candidates = article.find_all(["section", "div"])
        best = max(candidates, key=lambda t: len(t.get_text(strip=True)), default=None)
        if best:
            md = markdownify.markdownify(str(best))
            md = re.sub(r"\n{3,}", "\n\n", md).strip()

    if math_tokens:
        # An inline formula's source markup is often wrapped in newlines that
        # markdownify keeps (and that may be the only gap to the next word);
        # the sentence runs through it, so make them spaces -- or nothing
        # before punctuation.
        def _restore(m: "re.Match") -> str:
            before, token, after, next_char = m.groups()
            tex = math_tokens.get(token, token)
            if not tex.startswith("$"):
                return (before or "") + tex + (after or "")
            return (
                (" " if before else "")
                + tex
                + (("" if next_char in ",.;:)]" else " ") if after else "")
            )

        md = re.sub(
            r"(?:(?<=[^\n])(\n[ \t]*))?(MATHTOKEN\d{5}X)(?:(\n[ \t]*)(?=([^\n])))?",
            _restore,
            md,
        )
        md = re.sub(r"\n{3,}", "\n\n", md).strip()

    return md


# ---------------------------------------------------------------------------
# Fallback: Semantic Scholar abstract
# ---------------------------------------------------------------------------

def resolve_to_paper_id(identifier: str) -> Optional[str]:
    """
    Resolve any identifier (DOI or paperId) to a Semantic Scholar paperId.

    The SS Graph API accepts both bare paperIds and DOI:{doi} as the ID path segment.
    This function always calls SS Graph API regardless of input format, so no
    fragile string-based detection is needed.
    """
    # Try as-is first (bare paperId)
    url = f"{SS_GRAPH_API}/{identifier}"
    try:
        resp = requests.get(
            url,
            params={"fields": "paperId"},
            headers=_ss_headers(),
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("paperId")
    except Exception:
        pass

    # If that failed and the identifier looks like a DOI, retry with DOI: prefix
    if identifier.startswith("10."):
        url = f"{SS_GRAPH_API}/DOI:{identifier}"
        try:
            resp = requests.get(
                url,
                params={"fields": "paperId"},
                headers=_ss_headers(),
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json().get("paperId")
        except Exception:
            pass

    return None


def fetch_abstract(paper_id: str) -> Optional[str]:
    """
    Fetch the abstract for a paper via the Semantic Scholar Graph API.
    paper_id must be a Semantic Scholar paperId (not a DOI).

    Returns an HTML-escaped string wrapped in <p> tags, or None if no abstract.
    """
    import html  # defer to avoid unused import in Task 2
    url = f"{SS_GRAPH_API}/{paper_id}"
    try:
        resp = requests.get(
            url,
            params={"fields": "abstract"},
            headers=_ss_headers(),
            timeout=15,
        )
        if resp.status_code != 200:
            log.warning("SS abstract API returned %s for %s", resp.status_code, paper_id)
            return None
        data = resp.json()
        abstract = data.get("abstract")
        if not abstract:
            return None
        escaped = html.escape(abstract)
        return f"<p>{escaped}</p>"
    except Exception as e:
        log.warning("SS abstract fetch failed for %s: %s", paper_id, e)
        return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def get_cache(identifier: str) -> Optional[dict]:
    """
    Look up cached HTML for `identifier` (DOI or paperId).
    Returns the row dict, or None if not found.
    """
    try:
        with papers.get_db_connection() as conn:
            row = conn.execute(
                "SELECT id, url, source, html, fetched_at, error_count "
                "FROM cached_html WHERE id = ?",
                (identifier,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "url": row[1], "source": row[2],
            "html": row[3], "fetched_at": row[4], "error_count": row[5],
        }
    except Exception as e:
        log.warning("Cache lookup failed for %s: %s", identifier, e)
        return None


def upsert_cache(identifier: str, url: str, source: str,
                  html: str, error_count: int = 0) -> None:
    """Insert or replace a cached HTML entry."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    try:
        with papers.get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO cached_html (id, url, source, html, fetched_at, error_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    url = excluded.url,
                    source = excluded.source,
                    html = excluded.html,
                    fetched_at = excluded.fetched_at,
                    error_count = excluded.error_count
                """,
                (identifier, url, source, html, fetched_at, error_count),
            )
            conn.commit()
    except Exception as e:
        log.warning("Cache upsert failed for %s: %s", identifier, e)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _build_result(
    status: str,
    cache_hit: bool,
    source: str,
    url: str,
    fetched_at: str,
    html: str = "",
    fallback_reason: str = "",
    error: str = "",
    route: str = "",
) -> str:
    """Build a structured result dict and serialize it to YAML.

    ``route`` ("direct" | "exit_node" | "cache") records which network
    served the page; omitted when empty (e.g. an error before any fetch).
    """
    result = {
        "status": status,
        "cache_hit": cache_hit,
        "source": source,
        "url": url,
        "fetched_at": fetched_at,
    }
    if route:
        result["route"] = route
    if html:
        result["html"] = html
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    if error:
        result["error"] = error
    return yaml.dump(result, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _is_stale_fallback_row(row: dict) -> bool:
    """True when `row` is an abstract-only fallback that is worth retrying
    now: the exit node is available and the row has failed fewer than
    `_MAX_FALLBACK_RETRIES_VIA_EXIT_NODE` times."""
    if row.get("source") != "fallback":
        return False
    if (row.get("error_count") or 0) >= _MAX_FALLBACK_RETRIES_VIA_EXIT_NODE:
        return False
    return bool(exit_node.availability().get("available"))


def orchestrate_fetch(identifier: str, force_refresh: bool = False) -> str:
    """
    Top-level orchestrator: resolve identifier, check cache, fetch, fallback.

    Returns a YAML string result.
    """
    # 1. Resolve identifier → (url, source)
    try:
        url, source = classify_identifier(identifier)
    except ValueError as e:
        return _build_result(
            status="error", cache_hit=False, source="", url="",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )

    # 2. Check cache (cache key = identifier as-is). A cached abstract-only
    #    fallback is treated as a miss while the exit node can still be tried
    #    (see _MAX_FALLBACK_RETRIES_VIA_EXIT_NODE).
    if not force_refresh:
        row = get_cache(identifier)
        if row and _is_stale_fallback_row(row):
            log.info(
                "cached fallback for %s (error_count=%s) retried: exit node %s available",
                identifier, row.get("error_count"), config.EXIT_NODE_NAME,
            )
            row = None
        if row:
            return _build_result(
                status="cache_hit",
                cache_hit=True,
                source=row["source"],
                url=row["url"],
                fetched_at=row["fetched_at"],
                html=row["html"],
                route="cache",
            )

    reasons: List[str] = []

    # 2b. arXiv keeps the full text at other URLs than the abstract page an
    #     arXiv id resolves to: the LaTeXML HTML, else the PDF.
    if source == "arXiv":
        found, arxiv_reasons = fetch_arxiv_full_text(url)
        if found:
            upsert_cache(identifier, found["url"], source, found["content"])
            return _build_result(
                status="ok",
                cache_hit=False,
                source=source,
                url=found["url"],
                fetched_at=datetime.now(timezone.utc).isoformat(),
                html=found["content"],
                route=found["route"],
            )
        reasons.extend(arxiv_reasons)

    # 3. Own network first, exit node last. Every attempt's HTML goes through
    #    extract_paper_content; fewer than _MIN_EXTRACTED_CONTENT_CHARS of
    #    article text (paywall, consent wall) or a bot-challenge page
    #    (`is_bot_challenge`, whatever its length) counts as a miss.
    #    a) plain requests on this host's network (spec §6/§8 Phase 2,
    #       finding F9) -- no exit-node retry yet;
    #    b) headless Chromium, also on this host's network -- passes JS bot
    #       challenges a plain GET cannot (IEEE's AWS WAF), so an
    #       institutional IP here is actually used;
    #    c) plain requests via the exit node, only once both came up empty.
    #    Selenium (opt-in, PUBLISHER_FETCH_BROWSER=1 -- "Failed to start
    #    Chrome" in 5 of 8 production calls) and the abstract follow.
    # Best page seen that has only the abstract (no access, or IEEE Early
    # Access without a PDF): returned as the fallback if nothing better
    # turns up -- it beats Semantic Scholar's bare abstract.
    abstract_only: List[str] = []

    def _usable(html_content: str, label: str) -> Optional[str]:
        if is_bot_challenge(html_content):
            reasons.append(f"{label} was served a bot-challenge page")
            return None
        content = extract_paper_content(html_content, source, url)
        if lacks_full_text(html_content, source):
            pdf_link = pdf_only_url(html_content, source)
            pdf = images.cached(pdf_link) if pdf_link else None
            if pdf is None:
                reasons.append(f"{label} returned the abstract only, no full text or PDF")
                if content and not abstract_only:
                    abstract_only.append(content)
                return None
            content = with_pdf_full_text(content, pdf_link, pdf)
        n = len(content.strip()) if content else 0
        if n >= _MIN_EXTRACTED_CONTENT_CHARS:
            return content
        reasons.append(f"{label} returned too little extractable content ({n} chars)")
        return None

    def _ok(content: str, route: str) -> str:
        upsert_cache(identifier, url, source, content)
        return _build_result(
            status="ok",
            cache_hit=False,
            source=source,
            url=url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            html=content,
            route=route,
        )

    try:
        html_content, _route = fetch_via_requests_routed(url, direct_only=True)
        content = _usable(html_content, "direct requests fetch")
        if content:
            return _ok(content, "direct")
    except RequestsFetchError as e:
        reasons.append(str(e))

    if _headless_browser_bin():
        try:
            html_content = fetch_via_headless_browser(
                url,
                figure_urls=lambda html: images.remote_image_urls(
                    extract_paper_content(html, source, url)
                ),
                pdf_url=lambda html: pdf_only_url(html, source),
            )
            content = _usable(html_content, "direct headless browser")
            if content:
                return _ok(content, "direct")
        except BrowserError as e:
            reasons.append(f"direct headless browser: {e}")

    if exit_node.availability()["available"]:
        try:
            html_content, route = fetch_via_requests_routed(url, force_exit_node=True)
            content = _usable(html_content, f"via exit node {config.EXIT_NODE_NAME}")
            if content:
                return _ok(content, route)
        except RequestsFetchError as e:
            reasons.append(str(e))

    fallback_reason = "; ".join(reasons)

    if _publisher_fetch_browser_enabled():
        try:
            html_content = fetch_via_dev_browser(url)
            content = extract_paper_content(html_content, source, url)
            upsert_cache(identifier, url, source, content)
            return _build_result(
                status="ok",
                cache_hit=False,
                source=source,
                url=url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                html=content,
            )
        except BrowserError as e:
            fallback_reason = f"Browser fetch failed: {e}"

    # 4. Fallback: the publisher's own abstract page, else Semantic Scholar's
    #    abstract. Cached as "fallback" either way, so it is retried later
    #    (see _is_stale_fallback_row).
    if abstract_only:
        existing = get_cache(identifier)
        prev_errors = existing["error_count"] if existing else 0
        upsert_cache(identifier, url, "fallback", abstract_only[0],
                     error_count=prev_errors + 1)
        return _build_result(
            status="fallback",
            cache_hit=False,
            source="fallback",
            url=url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            html=abstract_only[0],
            fallback_reason=fallback_reason,
        )

    paper_id = resolve_to_paper_id(identifier)
    abstract_md = fetch_abstract(paper_id) if paper_id else None
    if abstract_md:
        existing = get_cache(identifier)
        prev_errors = existing["error_count"] if existing else 0
        upsert_cache(identifier, url, "fallback", abstract_md,
                      error_count=prev_errors + 1)
        return _build_result(
            status="fallback",
            cache_hit=False,
            source="fallback",
            url=url,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            html=abstract_md,
            fallback_reason=fallback_reason,
        )

    # 5. Total failure — do NOT cache
    return _build_result(
        status="error",
        cache_hit=False,
        source=source,
        url=url,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        error=f"{fallback_reason} | Abstract fallback also failed.",
    )
