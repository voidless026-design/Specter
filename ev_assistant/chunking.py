"""Structure-aware chunking - turns a document into retrievable pieces.

Naive fixed-width chunking cuts sentences in half and throws away the fact
that a paragraph sat under "Water > Purification". Both cost recall. This
module splits on structure first (headings, then paragraphs, then sentences)
and carries each chunk's place in the document along with it.

Two things here are deliberate and worth knowing about:

- Every chunk's stored text starts with a one-line context header,
  ``"{title} - {heading_path}"``. It costs a handful of tokens and measurably
  helps both the embedder and the keyword index, because a chunk that says
  "boil it for a minute" is otherwise about nothing in particular.
- Tables, lists and fenced code are atomic. A half-table retrieves worse than
  an oversized one, so they are never split even when they blow the budget.

Source formats seen in practice by `ev learn`: Wikipedia plain-text extracts
(``== Section ==`` headings), markdown (``## Section``), stripped HTML (no
headings at all) and source files (chunked by definition instead).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Same rough conversion knowledge.py uses: English averages ~4 chars/token.
# Close enough for chunk budgets, and it keeps a tokenizer out of the deps.
CHARS_PER_TOKEN = 4

TARGET_TOKENS = 500       # the 400-600 band, aimed at the middle
OVERLAP_RATIO = 0.15
MAX_TOKENS = 600          # a hard ceiling for prose; atomic blocks are exempt

HEADING_SEPARATOR = " > "

# Suffixes routed to the code chunker. Everything else is treated as prose.
CODE_SUFFIXES = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".cs": "c_sharp", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".scala": "scala", ".lua": "lua", ".sh": "bash", ".bash": "bash",
    ".sql": "sql",
}


@dataclass
class TextChunk:
    """One retrievable piece. `text` is what gets stored and embedded."""

    ordinal: int
    text: str            # context header + body
    body: str            # the source text alone, no header
    heading_path: str
    token_count: int

    def as_row(self) -> dict:
        """The dict shape `Store.add_document` expects."""
        return {
            "ordinal": self.ordinal,
            "text": self.text,
            "heading_path": self.heading_path,
            "token_count": self.token_count,
        }


def estimate_tokens(text: str) -> int:
    """Cheap token estimate. Deliberately approximate - it sizes budgets."""
    return max(1, len(text or "") // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------

# Words whose trailing full stop is not the end of a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon",
    "vs", "etc", "al", "fig", "figs", "no", "nos", "approx", "est",
    "inc", "ltd", "co", "corp", "dept", "univ", "ave", "blvd",
    "vol", "pp", "ed", "eds", "cf", "ca", "circa", "min", "max",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_BOUNDARY_RE = re.compile(r'([.!?]+["\'’”)\]]*)(\s+)')
_LAST_WORD_RE = re.compile(r'([A-Za-z][A-Za-z.]*)[.!?]+["\'’”)\]]*$')


def _ends_with_abbreviation(candidate: str) -> bool:
    m = _LAST_WORD_RE.search(candidate)
    if not m:
        return False
    word = m.group(1).rstrip(".")
    # A lone capital is an initial ("J. R. R. Tolkien"); a lone lowercase
    # letter is just a short word ending a sentence.
    if len(word) == 1:
        return word.isupper()
    # "U.S.", "e.g.", "a.m." - an internal dot means an abbreviation.
    return "." in word or word.lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split into sentences, leaving abbreviations and initials intact.

    A tokenizer-free approximation. It errs toward *not* splitting, because
    an over-long sentence is a smaller problem than a sentence cut in half.
    """
    text = (text or "").strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        candidate = text[start:m.end(1)]
        if _ends_with_abbreviation(candidate):
            continue
        stripped = candidate.strip()
        if stripped:
            sentences.append(stripped)
        start = m.end(2)
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------

# "## Section" (markdown) and "== Section ==" (Wikipedia plain-text extracts).
_ATX_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_WIKI_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")
# Setext underlines, "=" only: "---" is too easily a table rule or front matter.
_SETEXT_RE = re.compile(r"^={3,}$")

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_LIST_RE = re.compile(r"^\s*([-*+•]|\d+[.)])\s+\S")
_TABLE_RE = re.compile(r"^\s*\|")


@dataclass
class Block:
    """A paragraph-or-larger unit of the document, with its heading path."""

    text: str
    heading_path: str
    atomic: bool = False   # tables, lists, fenced code: never split

    @property
    def chars(self) -> int:
        # Budgets are tracked in characters, not estimated tokens: per-piece
        # token estimates round down and the error compounds, so a chunk
        # assembled from 40 "15-token" sentences comes out well over budget.
        return len(self.text)


def _heading_at(lines: list[str], i: int) -> tuple[str, int, int] | None:
    """(title, level, lines consumed) if line `i` opens a heading, else None."""
    line = lines[i]
    m = _WIKI_RE.match(line)
    if m:
        # "== X ==" is a top-level section in wikitext, so level 1.
        return m.group(2).strip(), len(m.group(1)) - 1, 1
    m = _ATX_RE.match(line)
    if m:
        return m.group(2).strip(), len(m.group(1)), 1
    text = line.strip()
    if text and i + 1 < len(lines) and _SETEXT_RE.match(lines[i + 1].strip()):
        if not _LIST_RE.match(line) and not _TABLE_RE.match(line):
            return text, 1, 2  # the title line plus its underline
    return None


def parse_blocks(text: str) -> list[Block]:
    """Break a document into heading-tagged blocks, atomic ones marked."""
    lines = (text or "").replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    stack: list[str] = []           # heading titles by level, index 0 = level 1
    buffer: list[str] = []
    buffer_kind = "para"            # para | list | table
    i = 0

    def path() -> str:
        return HEADING_SEPARATOR.join(h for h in stack if h)

    def flush() -> None:
        nonlocal buffer, buffer_kind
        body = "\n".join(buffer).strip()
        if body:
            blocks.append(Block(text=body, heading_path=path(), atomic=buffer_kind != "para"))
        buffer = []
        buffer_kind = "para"

    while i < len(lines):
        line = lines[i]

        heading = _heading_at(lines, i)
        if heading is not None:
            flush()
            title, level, consumed = heading
            del stack[level - 1:]
            while len(stack) < level - 1:
                stack.append("")   # a jump from h1 to h3 leaves a gap
            stack.append(title)
            i += consumed
            continue

        fence_match = _FENCE_RE.match(line)
        if fence_match:
            flush()
            fence = fence_match.group(1)
            fenced = [line]
            i += 1
            while i < len(lines):
                fenced.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            blocks.append(Block(text="\n".join(fenced).strip(), heading_path=path(), atomic=True))
            continue

        if not line.strip():
            flush()
            i += 1
            continue

        kind = "table" if _TABLE_RE.match(line) else "list" if _LIST_RE.match(line) else "para"
        # A continuation line inside a list or table keeps that block's kind.
        if buffer and kind == "para" and buffer_kind in ("list", "table"):
            kind = buffer_kind
        if buffer and kind != buffer_kind:
            flush()
        buffer_kind = kind
        buffer.append(line)
        i += 1

    flush()
    return blocks


# ---------------------------------------------------------------------------
# Prose chunking
# ---------------------------------------------------------------------------


def context_header(title: str, heading_path: str) -> str:
    """The one-line orientation prefix stored with every chunk."""
    title = (title or "").strip()
    heading_path = (heading_path or "").strip()
    if title and heading_path:
        return f"{title} - {heading_path}"
    return title or heading_path


def _with_header(title: str, heading_path: str, body: str) -> str:
    header = context_header(title, heading_path)
    return f"{header}\n\n{body}" if header else body


def _overlap_tail(body: str, max_chars: int) -> str:
    """The last whole sentences of `body`, up to `max_chars`.

    Whole sentences only - an overlap that starts mid-clause reads as noise
    to the embedder and defeats the point of having it. Capped at half the
    previous body so a short chunk isn't simply repeated wholesale.
    """
    budget = min(max_chars, len(body) // 2)
    if budget <= 0:
        return ""
    sentences = split_sentences(body)
    if len(sentences) < 2:
        return ""
    picked: list[str] = []
    for sentence in reversed(sentences):
        cost = len(sentence) + 1
        if picked and cost > budget:
            break
        picked.insert(0, sentence)
        budget -= cost
        if budget <= 0:
            break
    return " ".join(picked)


def _wrap_words(sentence: str, budget_chars: int) -> list[str]:
    """Last resort for a single sentence longer than a whole chunk.

    Sentence boundaries are a heuristic, so the occasional run-on slips
    through - usually stripped HTML with the punctuation lost. Splitting one
    at a word boundary beats emitting a chunk several times over budget.
    """
    if len(sentence) <= budget_chars:
        return [sentence]
    out: list[str] = []
    buf: list[str] = []
    buf_chars = 0
    for word in sentence.split():
        cost = len(word) + 1
        if buf and buf_chars + cost > budget_chars:
            out.append(" ".join(buf))
            buf, buf_chars = [], 0
        buf.append(word)
        buf_chars += cost
    if buf:
        out.append(" ".join(buf))
    return out


def _split_oversized(block: Block, budget_chars: int) -> list[Block]:
    """Sentence-split a too-long prose block. Atomic blocks pass through."""
    if block.atomic or block.chars <= budget_chars:
        return [block]
    out: list[Block] = []
    buf: list[str] = []
    buf_chars = 0
    sentences = split_sentences(block.text) or [block.text]
    for sentence in [p for s in sentences for p in _wrap_words(s, budget_chars)]:
        cost = len(sentence) + 1
        if buf and buf_chars + cost > budget_chars:
            out.append(Block(" ".join(buf), block.heading_path))
            buf, buf_chars = [], 0
        buf.append(sentence)
        buf_chars += cost
    if buf:
        out.append(Block(" ".join(buf), block.heading_path))
    return out


def _dominant_heading(blocks: list[Block]) -> str:
    """The heading path contributing the most text to a chunk.

    A chunk that absorbed a stub section spans two headings; labelling it by
    whichever one it is mostly made of beats labelling it by whichever one
    happened to come first.
    """
    weights: dict[str, int] = {}
    for block in blocks:
        weights[block.heading_path] = weights.get(block.heading_path, 0) + block.chars
    first = blocks[0].heading_path
    return max(weights, key=lambda path: (weights[path], path == first))


def chunk_prose(
    text: str,
    *,
    title: str = "",
    target_tokens: int = TARGET_TOKENS,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[TextChunk]:
    """Chunk prose on headings, then paragraphs, then sentences."""
    max_chars = max(target_tokens, int(target_tokens * 1.2)) * CHARS_PER_TOKEN
    overlap_chars = int(target_tokens * max(0.0, overlap_ratio)) * CHARS_PER_TOKEN
    # A stub section shorter than this is absorbed into the next one instead
    # of becoming a chunk of its own.
    merge_below = max(1, target_tokens // 4) * CHARS_PER_TOKEN

    raw = parse_blocks(text)
    if not raw:
        return []
    # A chunk is header + overlap + an absorbed stub + body, so the body only
    # gets what is left. Budgeting for all four is what keeps `max_chars` a
    # real ceiling rather than a suggestion. Atomic blocks are the documented
    # exception: they are never split, so a huge table still overshoots.
    header_chars = len(title) + 3 + max(len(b.heading_path) for b in raw) + 2
    body_budget = max(
        target_tokens * 2, max_chars - header_chars - overlap_chars - merge_below
    )

    blocks: list[Block] = []
    for block in raw:
        blocks.extend(_split_oversized(block, body_budget))

    chunks: list[TextChunk] = []
    buf: list[Block] = []
    buf_chars = 0

    def flush() -> None:
        nonlocal buf, buf_chars
        if not buf:
            return
        body = "\n\n".join(b.text for b in buf).strip()
        heading_path = _dominant_heading(buf)
        # Overlap carries the tail of the previous chunk, unless that chunk
        # was a single atomic block (repeating half a table helps nobody).
        if chunks and overlap_chars and not (len(buf) == 1 and buf[0].atomic):
            tail = _overlap_tail(chunks[-1].body, overlap_chars)
            if tail and tail not in body:
                body = f"{tail}\n\n{body}"
        stored = _with_header(title, heading_path, body)
        chunks.append(TextChunk(
            ordinal=len(chunks), text=stored, body=body,
            heading_path=heading_path, token_count=estimate_tokens(stored),
        ))
        buf, buf_chars = [], 0

    for block in blocks:
        cost = block.chars + 2  # the "\n\n" that will join it
        # A heading change is the preferred split point. The exception is a
        # stub section: alone it would be a tiny, context-free chunk, so it
        # rides along with the next one - which is also why both flushes wait
        # for the buffer to clear `merge_below`.
        if buf and buf_chars >= merge_below and block.heading_path != buf[-1].heading_path:
            flush()
        if buf and buf_chars >= merge_below and buf_chars + cost > body_budget:
            flush()
        buf.append(block)
        buf_chars += cost

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------

# tree-sitter node types that are worth a chunk of their own.
_DEFINITION_TYPES = {
    "function_definition", "function_declaration", "function_item",
    "method_definition", "method_declaration",
    "class_definition", "class_declaration", "class_specifier",
    "struct_item", "struct_specifier", "impl_item", "trait_item",
    "enum_item", "enum_declaration", "enum_specifier",
    "interface_declaration", "module", "mod_item",
    "type_alias_declaration", "abstract_class_declaration",
    "decorated_definition", "export_statement", "lexical_declaration",
}

# Fallback: a definition opening at column 0, for the common languages.
_DEF_LINE_RE = re.compile(
    r"^(?:(?:export|public|private|protected|internal)\s+)*"
    r"(?:(?:static|final|abstract|async|pub|const)\s+)*"
    r"(?:def|class|func|fn|function|impl|trait|struct|enum|interface|module|package|type)"
    r"\s+([A-Za-z_][\w.]*)"
)
_DECORATOR_RE = re.compile(r"^[@#\[]")


@dataclass
class Span:
    """A top-level definition: the lines it covers and what it's called."""

    start: int
    end: int
    name: str = ""


def detect_language(source_uri: str) -> str | None:
    """Map a path/URI to a tree-sitter language name, or None for prose."""
    return CODE_SUFFIXES.get(Path((source_uri or "").split("?")[0]).suffix.lower())


_NAME_CARRYING_TYPES = ("identifier", "type_identifier", "field_identifier", "constant")


def _node_name(node, depth: int = 0) -> str:
    """The declared name of a tree-sitter definition node, if it has one.

    Grammars disagree about where the name lives: Python and Go expose a
    `name` field, C buries it inside a `declarator`, and a Python decorator
    wraps the definition entirely. Try each in turn, then give up - an
    unnamed chunk falls back to a line marker, which is no worse than before.
    """
    if depth > 4:
        return ""
    try:
        for field in ("name", "declarator"):
            child = node.child_by_field_name(field)
            if child is None:
                continue
            if child.type in _NAME_CARRYING_TYPES and child.text:
                return child.text.decode("utf-8", "replace")
            found = _node_name(child, depth + 1)
            if found:
                return found
        for child in node.children:
            if child.type in _DEFINITION_TYPES:  # decorated_definition et al
                found = _node_name(child, depth + 1)
                if found:
                    return found
            if child.type in _NAME_CARRYING_TYPES and child.text:
                return child.text.decode("utf-8", "replace")
    except (AttributeError, UnicodeDecodeError):
        pass
    return ""


def _tree_sitter_spans(code: str, language: str) -> list[Span] | None:
    """Top-level definition spans, or None when tree-sitter can't help.

    tree-sitter is optional - it is not a runtime dependency - so this returns
    None whenever the parser or the grammar isn't installed and the caller
    falls back to the line-based splitter.
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None
    try:
        parser = get_parser(language)
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        logger.debug("tree-sitter couldn't parse %s, using the line splitter", language)
        return None
    spans = [
        Span(node.start_point[0], node.end_point[0], _node_name(node))
        for node in tree.root_node.children
        if node.type in _DEFINITION_TYPES
    ]
    return spans or None


def _definition_spans(lines: list[str]) -> list[Span]:
    """Definitions found by regex at column 0, when tree-sitter isn't there."""
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line.strip() or line[:1].isspace():
            continue
        m = _DEF_LINE_RE.match(line)
        if not m:
            continue
        start = i
        # Walk back over decorators/attributes so they stay with their target.
        while start > 0 and _DECORATOR_RE.match(lines[start - 1] or ""):
            start -= 1
        starts.append((start, m.group(1)))
    spans = []
    for n, (start, name) in enumerate(starts):
        end = (starts[n + 1][0] - 1) if n + 1 < len(starts) else len(lines) - 1
        spans.append(Span(start, end, name))
    return spans


def chunk_code(
    code: str,
    *,
    title: str = "",
    language: str | None = None,
    target_tokens: int = TARGET_TOKENS,
) -> list[TextChunk]:
    """Chunk source by definition boundaries, falling back to line windows.

    Definitions are never split: half a function is a chunk that answers
    nothing, so an oversized one is emitted whole, the same exemption tables
    and lists get in prose.
    """
    text = (code or "").replace("\r\n", "\n")
    if not text.strip():
        return []
    lines = text.split("\n")
    max_chars = max(target_tokens, int(target_tokens * 1.2)) * CHARS_PER_TOKEN
    header_chars = len(title) + 32
    budget = max(target_tokens * 2, max_chars - header_chars)

    spans = _tree_sitter_spans(text, language) if language else None
    used = "tree-sitter"
    if not spans:
        spans = _definition_spans(lines)
        used = "definition lines"
    if not spans:
        spans = [Span(i, min(i + 59, len(lines) - 1)) for i in range(0, len(lines), 60)]
        used = "line windows"
    logger.debug("chunking %s by %s (%d spans)", title or "code", used, len(spans))

    # Anything between definitions (imports, module docstring, trailing code)
    # becomes its own unnamed region rather than being dropped.
    regions: list[Span] = []
    cursor = 0
    for span in spans:
        if span.start > cursor and "\n".join(lines[cursor:span.start]).strip():
            regions.append(Span(cursor, span.start - 1))
        regions.append(span)
        cursor = span.end + 1
    if cursor < len(lines) and "\n".join(lines[cursor:]).strip():
        regions.append(Span(cursor, len(lines) - 1))

    chunks: list[TextChunk] = []
    buf: list[str] = []
    buf_chars = 0
    buf_start = 0
    buf_names: list[str] = []

    def flush() -> None:
        nonlocal buf, buf_chars, buf_names
        body = "\n".join(buf).strip("\n")
        if body.strip():
            heading_path = HEADING_SEPARATOR.join(buf_names) if buf_names else f"L{buf_start + 1}"
            stored = _with_header(title, heading_path, body)
            chunks.append(TextChunk(
                ordinal=len(chunks), text=stored, body=body,
                heading_path=heading_path, token_count=estimate_tokens(stored),
            ))
        buf, buf_chars, buf_names = [], 0, []

    for region in regions:
        block = lines[region.start:region.end + 1]
        cost = len("\n".join(block)) + 1
        if buf and buf_chars + cost > budget:
            flush()
        if not buf:
            buf_start = region.start
        buf.extend(block)
        buf_chars += cost
        if region.name:
            buf_names.append(region.name)
        if buf_chars >= budget:
            flush()
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def chunk_document(
    text: str,
    *,
    title: str = "",
    source_uri: str = "",
    source_type: str = "",
    target_tokens: int = TARGET_TOKENS,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[TextChunk]:
    """Chunk a document, picking the prose or code strategy from its source."""
    language = detect_language(source_uri)
    if source_type == "code" or language:
        return chunk_code(text, title=title, language=language, target_tokens=target_tokens)
    return chunk_prose(
        text, title=title, target_tokens=target_tokens, overlap_ratio=overlap_ratio
    )


def chunk_rows(text: str, **kwargs) -> list[dict]:
    """`chunk_document` in the dict shape `Store.add_document` takes."""
    return [c.as_row() for c in chunk_document(text, **kwargs)]
