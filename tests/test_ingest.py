from __future__ import annotations

import pytest

from ev_assistant import ingest
from ev_assistant.ingest import Fetched, extract_links, is_safe_url, resolve


# ---- source sniffing ----

def test_resolve_detects_url():
    assert resolve("https://example.com/page") == "url"
    assert resolve("http://example.com") == "url"
    assert resolve("example.com/guide") == "url"


def test_resolve_detects_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("hello", encoding="utf-8")
    assert resolve(str(f)) == "file"


def test_resolve_falls_back_to_wikipedia():
    assert resolve("Water purification") == "wikipedia"
    assert resolve("how to build a shelter") == "wikipedia"


def test_resolve_rejects_empty():
    with pytest.raises(ValueError):
        resolve("   ")


# ---- URL safety ----

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/admin",
    "https://localhost/x",
    "http://192.168.1.5/",
    "http://10.0.0.1/",
    "http://169.254.169.254/latest/meta-data",  # cloud metadata
    "file:///etc/passwd",
    "ftp://example.com",
    "http://box.local/",
])
def test_unsafe_urls_rejected(url):
    assert is_safe_url(url) is False


@pytest.mark.parametrize("url", ["https://example.com", "http://en.wikipedia.org/wiki/Fire"])
def test_safe_urls_allowed(url):
    assert is_safe_url(url) is True


# ---- link extraction ----

HTML = """
<html><body>
  <a href="/page2">two</a>
  <a href="https://example.com/page3">three</a>
  <a href="https://other.com/x">offsite</a>
  <a href="#frag">frag</a>
  <a href="mailto:a@b.c">mail</a>
  <a href="http://127.0.0.1/secret">local</a>
  <a href="/page2">dupe</a>
</body></html>
"""


def test_extract_links_same_domain_only():
    links = extract_links(HTML, "https://example.com/start", same_domain=True)
    assert "https://example.com/page2" in links
    assert "https://example.com/page3" in links
    assert all("other.com" not in ln for ln in links)
    assert all("127.0.0.1" not in ln for ln in links)
    assert len(links) == len(set(links))  # de-duplicated


def test_extract_links_any_domain():
    links = extract_links(HTML, "https://example.com/start", same_domain=False)
    assert any("other.com" in ln for ln in links)


def test_extract_links_skips_fragments_and_mailto():
    links = extract_links(HTML, "https://example.com/start", same_domain=False)
    assert all(not ln.startswith("mailto:") for ln in links)
    assert all("#" not in ln for ln in links)


# ---- crawl bounding ----

def _fake_site(monkeypatch, pages: dict[str, list[str]]):
    """fetch() stub: each source yields its listed links."""
    def fake_fetch(source, want_links=False, same_domain=True):
        if source not in pages:
            raise ValueError("404")
        return Fetched(title=source, text=f"body of {source}", source=source,
                       links=pages[source] if want_links else [])
    monkeypatch.setattr(ingest, "fetch", fake_fetch)


def test_crawl_depth_zero_fetches_only_the_seed(monkeypatch):
    _fake_site(monkeypatch, {"a": ["b", "c"], "b": [], "c": []})
    docs = list(ingest.crawl("a", depth=0))
    assert [d.source for d in docs] == ["a"]


def test_crawl_follows_links_to_depth(monkeypatch):
    _fake_site(monkeypatch, {"a": ["b", "c"], "b": ["d"], "c": [], "d": []})
    docs = list(ingest.crawl("a", depth=1))
    assert [d.source for d in docs] == ["a", "b", "c"]  # d is depth 2, excluded


def test_crawl_respects_max_pages(monkeypatch):
    _fake_site(monkeypatch, {"a": ["b", "c", "d"], "b": [], "c": [], "d": []})
    docs = list(ingest.crawl("a", depth=1, max_pages=2))
    assert len(docs) == 2


def test_crawl_terminates_on_cycles(monkeypatch):
    _fake_site(monkeypatch, {"a": ["b"], "b": ["a"]})
    docs = list(ingest.crawl("a", depth=5, max_pages=10))
    assert [d.source for d in docs] == ["a", "b"]  # visited set stops the loop


def test_crawl_skips_pages_that_fail(monkeypatch):
    _fake_site(monkeypatch, {"a": ["missing", "c"], "c": []})
    docs = list(ingest.crawl("a", depth=1))
    assert [d.source for d in docs] == ["a", "c"]
