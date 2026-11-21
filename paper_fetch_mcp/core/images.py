"""Paper figures and PDFs: an on-disk cache for assets fetched along with an
online paper, and copying a paper's images (and a linked full-text PDF) next
to a saved copy of it.

Why a cache: some publishers only serve figures to the browser session that
passed their bot challenge (IEEE Xplore answers a plain GET of
``/mediastore/...gif`` with 403). `core.publisher`'s headless-browser fetch
therefore downloads the article's figures from inside that session and
`store()`s them here -- as it does a PDF-only paper's full-text PDF (IEEE
Early Access) -- and `localize()` later copies them out of the cache when
`get_paper(..., save_path=...)` writes the paper to disk, falling back to a
plain download for images that were never cached (publishers without a
challenge).

Why a folder and not inline ``data:`` URIs: saved papers are read back by
agents; a dozen base64 figures would put megabytes of noise into the very
file they read. ``<stem>_files/`` next to ``<stem>.md`` with relative links
renders in every Markdown viewer (VS Code, Obsidian, GitHub, pandoc).

No side effects at import time (core/__init__.py rule): the cache directory
is created on first `store()`.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests

from paper_fetch_mcp import config
from paper_fetch_mcp.core import exit_node

logger = logging.getLogger(__name__)

# ![alt](target) -- the target ends at the first space or ")" (an optional
# "title" part after a space is left in place).
IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)")

_DOWNLOAD_TIMEOUT_S = 15
_MAX_ASSET_BYTES = 80 * 1024 * 1024
_IMAGE_EXTS = (".gif", ".png", ".jpg", ".jpeg", ".webp", ".svg", ".bmp", ".tif", ".tiff")
_ASSET_EXTS = _IMAGE_EXTS + (".pdf",)
# [text](target) that is NOT an image -- a candidate attachment (cached PDF).
LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)")


def image_targets(markdown: str) -> List[str]:
    """Every distinct image link target in `markdown`, in order."""
    return list(dict.fromkeys(m.group(2) for m in IMAGE_LINK_RE.finditer(markdown or "")))


def remote_image_urls(markdown: str) -> List[str]:
    return [t for t in image_targets(markdown) if t.startswith(("http://", "https://"))]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _ext(url: str, mime: str = "") -> str:
    guessed = mimetypes.guess_extension((mime or "").split(";")[0].strip()) or ""
    if guessed in _ASSET_EXTS:
        return guessed
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix if suffix in _ASSET_EXTS else ".img"


def _key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]


def store(url: str, data: bytes, mime: str = "") -> Optional[Path]:
    """Cache `data` as the asset (image or PDF) at `url`; returns the cached
    file, or None if it was empty, oversized, or could not be written."""
    if not data or len(data) > _MAX_ASSET_BYTES:
        return None
    cache_dir = Path(config.PUBLISHER_ASSET_CACHE_DIR)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{_key(url)}{_ext(url, mime)}"
        path.write_bytes(data)
    except OSError as e:
        logger.warning("could not cache image %s: %s", url, e)
        return None
    return path


def cached(url: str) -> Optional[Path]:
    """The cached file for `url`, if any."""
    cache_dir = Path(config.PUBLISHER_ASSET_CACHE_DIR)
    if not cache_dir.is_dir():
        return None
    return next(iter(sorted(cache_dir.glob(f"{_key(url)}.*"))), None)


def download(
    url: str, kinds: Tuple[str, ...] = ("image/",), timeout: float = _DOWNLOAD_TIMEOUT_S
) -> Optional[Path]:
    """Fetch `url` with a plain GET (own network, then the exit node on an
    access failure) and cache it. Returns None unless the response's type
    starts with one of `kinds` (default: an image)."""
    try:
        routed = exit_node.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                   "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                     "Accept": "image/*,application/pdf;q=0.9,*/*;q=0.8"},
            timeout=timeout,
        )
    except requests.RequestException as e:
        logger.info("download of %s failed: %s", url, e)
        return None
    resp = routed.response
    mime = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or not mime.startswith(kinds):
        logger.info("download of %s: HTTP %s %s", url, resp.status_code, mime)
        return None
    return store(url, resp.content, mime)


# ---------------------------------------------------------------------------
# Localising a paper's images next to a saved copy
# ---------------------------------------------------------------------------


def _dest_name(target: str, taken: set) -> str:
    raw = Path(urlsplit(target).path).name or "image"
    stem, suffix = Path(raw).stem, Path(raw).suffix
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80] or "image"
    name, n = f"{stem}{suffix}", 1
    while name in taken:
        n += 1
        name = f"{stem}_{n}{suffix}"
    taken.add(name)
    return name


def localize(
    markdown: str,
    md_path: Path,
    *,
    local_files: Optional[Dict[str, Path]] = None,
    download_missing: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Copy the images `markdown` links to into ``<md stem>_files/`` beside
    `md_path`, and a linked full-text PDF that is in the cache to
    ``<md stem>.pdf`` beside it; rewrite those links to point there
    (relatively).

    A remote (http/https) image comes from the cache, else -- with
    `download_missing` -- a plain download. A relative target (corpus
    papers link ``chunk_0_img-0.jpg``) is looked up by file name in
    `local_files`. Targets that cannot be resolved keep their original link.
    A PDF is only ever copied from the cache (it was fetched with the page).

    Returns ``(markdown, report)`` with ``report = {saved, failed, dir?,
    pdf?}`` (``failed`` lists up to 10 unresolved image targets; ``dir``
    only when an image was copied; ``pdf`` the copied PDF's path), or ``{}``
    when the markdown links no images and no cached PDF.
    """
    targets = image_targets(markdown)
    pdfs = [
        t for t in dict.fromkeys(m.group(2) for m in LINK_RE.finditer(markdown or ""))
        if t.startswith(("http://", "https://"))
        and (cached(t) or Path()).suffix == ".pdf"
    ]
    if not targets and not pdfs:
        return markdown, {}

    files_dir = md_path.parent / f"{md_path.stem}_files"
    local_files = local_files or {}
    mapping: Dict[str, str] = {}
    failed: List[str] = []
    taken: set = set()

    for target in targets:
        if target.startswith(("http://", "https://")):
            src = cached(target) or (download(target) if download_missing else None)
        else:
            src = local_files.get(Path(target).name)
        if not src or not Path(src).is_file():
            failed.append(target)
            continue
        name = _dest_name(target, taken)
        if not Path(name).suffix:
            name += Path(src).suffix
        try:
            files_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, files_dir / name)
        except OSError as e:
            logger.warning("could not copy image %s -> %s: %s", src, files_dir / name, e)
            failed.append(target)
            continue
        mapping[target] = f"{files_dir.name}/{name}"

    pdf_mapping: Dict[str, str] = {}
    for i, target in enumerate(pdfs):
        dest = md_path.with_name(f"{md_path.stem}{'' if i == 0 else f'_{i + 1}'}.pdf")
        try:
            shutil.copyfile(cached(target), dest)
        except OSError as e:
            logger.warning("could not copy PDF %s -> %s: %s", target, dest, e)
            continue
        pdf_mapping[target] = dest.name

    def _swap(m: re.Match) -> str:
        new = mapping.get(m.group(2))
        return f"![{m.group(1)}]({new}" if new else m.group(0)

    def _swap_pdf(m: re.Match) -> str:
        new = pdf_mapping.get(m.group(2))
        return f"[{m.group(1)}]({new}" if new else m.group(0)

    report: Dict[str, Any] = {"saved": len(mapping), "failed": failed[:10]}
    if mapping:
        report["dir"] = str(files_dir)
    if pdf_mapping:
        report["pdf"] = str(md_path.with_name(next(iter(pdf_mapping.values()))))
    markdown = IMAGE_LINK_RE.sub(_swap, markdown)
    return LINK_RE.sub(_swap_pdf, markdown), report
