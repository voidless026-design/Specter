"""Fetch and extract text for `ev learn` - feeds E.V.'s offline knowledge base.

Sources: Wikipedia article titles, arbitrary web pages, and local text /
markdown / PDF files. Each returns (title, plain_text) which the caller
stores via Knowledge.add_document. Kept dependency-light: Wikipedia and web
pages use httpx + a simple HTML strip; PDFs use pypdf if it's installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_ANYTAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n\s*\n\s*")


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
