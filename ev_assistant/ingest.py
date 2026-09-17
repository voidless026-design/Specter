"""Fetch and extract text for `ev learn` - feeds E.V.'s offline knowledge base.

Sources: Wikipedia article titles, arbitrary web pages, and local text /
markdown / PDF files. `resolve()` sniffs which one you gave it, so callers
can just hand over a string. `fetch()` returns the text plus any links, and
`crawl()` walks those links to a bounded depth to build a corpus.

Kept dependency-light: Wikipedia and web pages use httpx + a simple HTML
strip; PDFs use pypdf if it's installed.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_ANYTAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n\s*")
_HREF_RE = re.compile(r"""<a\b[^>]*?href\s*=\s*["']([^"']+)["']""", re.IGNORECASE)

USER_AGENT = "ev-assistant/0.2"


@dataclass
class Fetched:
    """One retrieved document, plus links to follow when crawling."""

    title: str
    text: str
    source: str
    links: list[str] = field(default_factory=list)


def _strip_html(html: str) -> str:
    html = _TAG_RE.sub(" ", html)
    html = _ANYTAG_RE.sub(" ", html)
    # Unescape the few entities that matter for readability.
    for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        html = html.replace(entity, char)
    html = _WS_RE.sub(" ", html)
    html = _BLANKLINES_RE.sub("\n\n", html)
    return html.strip()


def from_wikipedia(title: str, lang: str = "en") -> tuple[str, str]:
    """Fetch a Wikipedia article's plain-text extract."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    resp = httpx.get(url, params=params, timeout=30, headers={"User-Agent": "ev-assistant/0.2"})
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        if extract:
            return page.get("title", title), extract
    raise ValueError(f"No Wikipedia article found for '{title}'.")


def from_url(url: str) -> tuple[str, str]:
    """Fetch a web page and return (title, readable text)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "https://" + url
    resp = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "ev-assistant/0.2"})
    resp.raise_for_status()
    html = resp.text
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = _strip_html(title_match.group(1)) if title_match else url
    return title, _strip_html(html)


def from_file(path: Path) -> tuple[str, str]:
    """Read a local .txt/.md/.pdf file and return (title, text)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path.stem, _read_pdf(path)
    if suffix in (".txt", ".md", ".markdown", ".rst", ""):
        return path.stem, path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type '{suffix}'. Use .txt, .md, or .pdf.")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("Reading PDFs needs pypdf: pip install pypdf") from e
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


# ---------------------------------------------------------------------------
# Source sniffing + safe fetching
# ---------------------------------------------------------------------------


def resolve(source: str) -> str:
    """Classify a source string: 'file', 'url', or 'wikipedia'.

    Lets `ev learn X` accept a path, a link, or a bare topic without flags.
    """
    s = source.strip()
    if not s:
        raise ValueError("Empty source.")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        return "url"
    if Path(s).expanduser().is_file():
        return "file"
    # Looks like a bare domain/path rather than a sentence.
    if re.match(r"^[\w.-]+\.[a-zA-Z]{2,}(/\S*)?$", s) and " " not in s:
        return "url"
    return "wikipedia"


def is_safe_url(url: str) -> bool:
    """Reject non-web schemes and anything pointing at the local machine.

    Crawling follows links from pages we don't control, so this keeps a
    stray link from making E.V. fetch internal/loopback addresses.
    """
    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ("localhost", "0.0.0.0") or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True  # a normal DNS name
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)


def extract_links(html: str, base_url: str, same_domain: bool = True) -> list[str]:
    """Absolute, de-duplicated, safe http(s) links found in `html`."""
    base_host = (urlparse(base_url).hostname or "").lower()
    out: list[str] = []
    seen = set()
    for href in _HREF_RE.findall(html):
        href = href.strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        absolute, _ = urldefrag(urljoin(base_url, href))
        if not is_safe_url(absolute):
            continue
        if same_domain and (urlparse(absolute).hostname or "").lower() != base_host:
            continue
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def wikipedia_links(title: str, lang: str = "en", limit: int = 25) -> list[str]:
    """Article titles linked from a Wikipedia page (for crawling)."""
    try:
        resp = httpx.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "links", "titles": title,
                    "plnamespace": 0, "pllimit": str(limit), "format": "json", "redirects": "1"},
            timeout=30, headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        return [ln["title"] for page in pages.values() for ln in page.get("links", [])]
    except Exception:
        logger.exception("Couldn't list Wikipedia links for %s", title)
        return []


def fetch(source: str, want_links: bool = False, same_domain: bool = True) -> Fetched:
    """Fetch any supported source, auto-detecting its kind."""
    kind = resolve(source)
    if kind == "file":
        title, text = from_file(Path(source).expanduser())
        return Fetched(title=title, text=text, source=source)
    if kind == "url":
        url = source if "://" in source else "https://" + source
        if not is_safe_url(url):
            raise ValueError(f"Refusing to fetch unsafe or local URL: {url}")
        resp = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        html = resp.text
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = _strip_html(m.group(1)) if m else url
        links = extract_links(html, str(resp.url), same_domain) if want_links else []
        return Fetched(title=title, text=_strip_html(html), source=url, links=links)
    title, text = from_wikipedia(source)
    links = wikipedia_links(title) if want_links else []
    return Fetched(title=title, text=text, source=f"wikipedia:{title}", links=links)


def crawl(
    source: str,
    depth: int = 0,
    max_pages: int = 20,
    same_domain: bool = True,
) -> Iterator[Fetched]:
    """Breadth-first fetch of `source` and its links, `depth` levels deep.

    Bounded on purpose: `max_pages` is a hard budget because one Wikipedia
    article two levels deep is thousands of pages. Already-visited sources
    are skipped, so cycles terminate.
    """
    queue: deque[tuple[str, int]] = deque([(source, depth)])
    visited: set[str] = set()
    fetched = 0

    while queue and fetched < max_pages:
        current, remaining = queue.popleft()
        key = current.strip().lower()
        if key in visited:
            continue
        visited.add(key)

        try:
            doc = fetch(current, want_links=remaining > 0, same_domain=same_domain)
        except Exception as e:
            logger.warning("Skipping %s: %s", current, e)
            continue

        fetched += 1
        yield doc

        if remaining > 0:
            for link in doc.links:
                if link.strip().lower() not in visited:
                    queue.append((link, remaining - 1))
