"""Tests for paper_fetch_mcp.core.images (figure cache + localising a saved
paper's images) and its use by core.papers.save_paper_content. No network:
downloads go through a patched `exit_node.get`."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from paper_fetch_mcp import config
from paper_fetch_mcp.core import images, papers

FIG1 = "https://ieeexplore.ieee.org/mediastore/x/fig1-large.gif"
FIG2 = "https://ieeexplore.ieee.org/mediastore/x/fig2-large.gif"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setattr(config, "PUBLISHER_ASSET_CACHE_DIR", d)
    return d


def _routed(status=200, content=b"GIF89a", ctype="image/gif"):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.content = content
    resp.headers = {"Content-Type": ctype}
    return MagicMock(response=resp)


def test_image_targets_in_order_without_duplicates():
    md = f"![a]({FIG1}) text ![b](rel/x.png) ![c]({FIG1}) ![d]({FIG2} \"title\")"
    assert images.image_targets(md) == [FIG1, "rel/x.png", FIG2]
    assert images.remote_image_urls(md) == [FIG1, FIG2]


def test_store_and_cached_round_trip(cache):
    assert images.cached(FIG1) is None
    path = images.store(FIG1, b"GIF89a-data", "image/gif")
    assert path.parent == cache and path.suffix == ".gif"
    assert images.cached(FIG1) == path and path.read_bytes() == b"GIF89a-data"
    # extension from the mime type when the URL has none
    assert images.store("https://x/figure?id=3", b"\x89PNG", "image/png").suffix == ".png"
    # empty data is not cached
    assert images.store(FIG2, b"", "image/gif") is None


def test_localize_copies_cached_figures_and_rewrites_links(tmp_path, cache):
    images.store(FIG1, b"GIF89a-1", "image/gif")
    md_path = tmp_path / "out" / "paper.md"
    md = f"Intro\n\n![Fig. 1]({FIG1})\n\n**Fig. 1.** caption"
    with patch("paper_fetch_mcp.core.images.exit_node.get") as get:
        new, report = images.localize(md, md_path)
    get.assert_not_called()
    assert "![Fig. 1](paper_files/fig1-large.gif)" in new
    assert (tmp_path / "out" / "paper_files" / "fig1-large.gif").read_bytes() == b"GIF89a-1"
    assert report == {"saved": 1, "failed": [], "dir": str(tmp_path / "out" / "paper_files")}


def test_localize_downloads_uncached_and_keeps_unobtainable_links(tmp_path, cache):
    md = f"![a]({FIG1})\n![b]({FIG2})"
    with patch(
        "paper_fetch_mcp.core.images.exit_node.get",
        side_effect=[_routed(content=b"GIF89a-dl"), _routed(403, b"<html>", "text/html")],
    ):
        new, report = images.localize(md, tmp_path / "p.md")
    assert "![a](p_files/fig1-large.gif)" in new
    assert f"![b]({FIG2})" in new  # 403 -> original link kept
    assert report["saved"] == 1 and report["failed"] == [FIG2]
    assert images.cached(FIG1).read_bytes() == b"GIF89a-dl"  # downloads are cached too


def test_localize_relative_corpus_figures_from_local_files(tmp_path, cache):
    src = tmp_path / "mistral" / "chunk_0_img-0.jpg"
    src.parent.mkdir()
    src.write_bytes(b"\xff\xd8jpeg")
    md = "![chunk_0_img-0.jpg](chunk_0_img-0.jpg) and ![x](missing.jpg)"
    new, report = images.localize(
        md, tmp_path / "k.md", local_files={"chunk_0_img-0.jpg": src}
    )
    assert "![chunk_0_img-0.jpg](k_files/chunk_0_img-0.jpg)" in new
    assert "![x](missing.jpg)" in new
    assert report["saved"] == 1 and report["failed"] == ["missing.jpg"]


def test_localize_dedupes_clashing_file_names(tmp_path, cache):
    a, b = "https://a.org/x/fig.png", "https://b.org/y/fig.png"
    images.store(a, b"A", "image/png")
    images.store(b, b"B", "image/png")
    new, _ = images.localize(f"![1]({a}) ![2]({b})", tmp_path / "p.md")
    assert "![1](p_files/fig.png)" in new and "![2](p_files/fig_2.png)" in new
    assert (tmp_path / "p_files" / "fig_2.png").read_bytes() == b"B"


def test_localize_without_images_is_a_no_op(tmp_path):
    assert images.localize("just text", tmp_path / "p.md") == ("just text", {})
    assert not (tmp_path / "p_files").exists()


def test_save_paper_content_reports_localized_images(tmp_path, cache):
    images.store(FIG1, b"GIF89a", "image/gif")
    report = papers.save_paper_content(f"![Fig. 1]({FIG1})", str(tmp_path / "x.md"))
    assert report["images"]["saved"] == 1
    assert (tmp_path / "x.md").read_text() == "![Fig. 1](x_files/fig1-large.gif)"
    assert report["chars"] == len("![Fig. 1](x_files/fig1-large.gif)")


def test_save_paper_content_without_images_has_no_images_block(tmp_path):
    report = papers.save_paper_content("plain text", str(tmp_path / "x.md"))
    assert "images" not in report and report["chars"] == len("plain text")


def test_localize_copies_a_cached_pdf_next_to_the_file(tmp_path, cache):
    pdf_url = "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=1&ref="
    images.store(pdf_url, b"%PDF-1.5", "application/pdf")
    md = f"# T\n\n[Full-text PDF]({pdf_url})\n\n[not cached](https://x.org/a.pdf)"
    new, report = images.localize(md, tmp_path / "paper.md")
    assert "[Full-text PDF](paper.pdf)" in new
    assert "[not cached](https://x.org/a.pdf)" in new
    assert (tmp_path / "paper.pdf").read_bytes() == b"%PDF-1.5"
    assert report["pdf"] == str(tmp_path / "paper.pdf") and report["saved"] == 0
