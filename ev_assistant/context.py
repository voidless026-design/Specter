"""Context assembly - turning retrieved chunks into the block the brain reads.

Three things here are not cosmetic:

**Ordering.** The highest-scoring chunks go at the *start and end* of the
block, with weaker material in the middle. Attention over a long context is
measurably strongest at the edges, so putting the best evidence in the middle
is throwing it away. The layout is a "V": best first, then descending, then
the second-best last.

**Source tags.** Every chunk is labelled ``[S1] Wikipedia - Tungsten`` so the
brain can cite where a claim came from, and so a person can ask "where did
you get that?" and get a real answer.

**Saying nothing, loudly.** When retrieval came back empty, the block is not
omitted - it says explicitly that the local store had nothing relevant. A
silent absence looks identical to a store that was never consulted, and the
brain needs to know the difference so it can say "that's from general
knowledge, not from your notes".

Truncation is always at a sentence boundary. A chunk cut mid-sentence reads
as though the source itself is unreliable.
"""

from __future__ import annotations

import logging

from ev_assistant.chunking import estimate_tokens, split_sentences
from ev_assistant.retrieval import BELOW_THRESHOLD, EMPTY_STORE, SKIPPED, RetrievalResult
from ev_assistant.store import Chunk, Fact

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_TOKENS = 4000
MIN_CHUNK_TOKENS = 40          # below this a truncated chunk isn't worth keeping
FACTS_BUDGET_RATIO = 0.25      # facts are compact; don't let them eat the block

HEADER = "RETRIEVED FROM YOUR KNOWLEDGE BASE"
FACTS_HEADER = "Facts on file:"

# What goes in when there is nothing to put in. Phrased so the brain knows
# the store was asked and had nothing, not that it was never consulted.
NOTHING_RELEVANT = (
    f"{HEADER}\n"
    "Nothing. The local knowledge base was searched and had no relevant material, "
    "so answer from your own knowledge and say plainly that this isn't from the "
    "user's saved notes or documents."
)
NOT_SEARCHED = ""   # a question that never needed the store gets no block at all


def source_label(chunk: Chunk) -> str:
    """The human half of a source tag: "Wikipedia - Tungsten"."""
    document = chunk.document
    if document is None:
        return "Unknown source"
    kind = (document.source_type or "").strip()
    pretty = {
        "wikipedia": "Wikipedia", "web": "Web", "pdf": "PDF", "epub": "Book",
        "code": "Code", "note": "Your notes", "file": "Your files",
        "conversation": "Earlier conversation", "voice": "Something you said",
        "feed": "News feed", "transcript": "Transcript",
    }.get(kind.lower(), kind.title() or "Source")
    title = (document.title or document.source_uri or "untitled").strip()
    return f"{pretty} - {title}"


def truncate_to_sentence(text: str, max_tokens: int) -> str:
    """Trim to a whole sentence within budget. Never a half sentence.

    Falls back to a word boundary only when the very first sentence is
    already over budget, which beats emitting a chunk that stops mid-word.
    """
    if estimate_tokens(text) <= max_tokens:
        return text
    kept: list[str] = []
    used = 0
    for sentence in split_sentences(text):
        cost = estimate_tokens(sentence)
        if kept and used + cost > max_tokens:
            break
        if not kept and cost > max_tokens:
            words = sentence.split()
            out: list[str] = []
            for word in words:
                if estimate_tokens(" ".join(out + [word])) > max_tokens:
                    break
                out.append(word)
            return (" ".join(out) + " ...").strip()
        kept.append(sentence)
        used += cost
    return " ".join(kept).strip()


def format_facts(facts: list[Fact], max_tokens: int) -> tuple[str, int]:
    """The compact fact list that sits above the prose. ("" , 0) if none fit."""
    if not facts or max_tokens <= 0:
        return "", 0
    lines: list[str] = []
    used = estimate_tokens(FACTS_HEADER)
    for fact in sorted(facts, key=lambda f: f.confidence, reverse=True):
        line = f"- {fact.statement.strip()}"
        cost = estimate_tokens(line)
        if used + cost > max_tokens:
            break
        lines.append(line)
        used += cost
    if not lines:
        return "", 0
    return f"{FACTS_HEADER}\n" + "\n".join(lines), used


def edge_order(items: list) -> list:
    """Reorder so the strongest material sits at both ends.

    Input is best-first. Output puts the best at the start, the second-best
    at the very end, and the rest descending in between - so nothing good is
    stranded in the middle, where attention is weakest.
    """
    if len(items) < 3:
        return list(items)
    head, tail = items[0], items[1]
    return [head, *items[2:], tail]


class ContextBlock:
    """The assembled block, plus what went into it."""

    def __init__(self, text: str = "", sources: list[dict] | None = None,
                 tokens: int = 0, reason: str = "", dropped: int = 0,
                 truncated: int = 0):
        self.text = text
        self.sources = sources or []
        self.tokens = tokens
        self.reason = reason
        self.dropped = dropped
        self.truncated = truncated

    def __bool__(self) -> bool:
        return bool(self.text)

    @property
    def cited(self) -> bool:
        return bool(self.sources)

    def describe(self) -> str:
        return (f"{len(self.sources)} sources, ~{self.tokens} tokens"
                f"{f', {self.dropped} dropped for budget' if self.dropped else ''}"
                f"{f', {self.truncated} truncated' if self.truncated else ''}")


def build(result: RetrievalResult, cfg=None) -> ContextBlock:
    """Assemble the context block handed to the brain."""
    budget = int(getattr(cfg, "context_budget_tokens", DEFAULT_BUDGET_TOKENS))

    if result.reason == SKIPPED:
        # Never asked, so there is nothing to report either way.
        return ContextBlock(text=NOT_SEARCHED, reason=SKIPPED)

    if not result.chunks and not result.facts:
        return ContextBlock(text=NOTHING_RELEVANT, reason=result.reason or EMPTY_STORE)

    facts_text, facts_tokens = format_facts(
        result.facts, int(budget * FACTS_BUDGET_RATIO)
    )
    remaining = budget - facts_tokens - estimate_tokens(HEADER)

    entries: list[dict] = []
    dropped = 0
    truncated = 0
    for index, chunk in enumerate(result.chunks, start=1):
        tag = f"[S{index}]"
        label = source_label(chunk)
        overhead = estimate_tokens(f"{tag} {label}\n")
        available = remaining - overhead
        if available < MIN_CHUNK_TOKENS:
            dropped = len(result.chunks) - index + 1
            break
        body = chunk.text
        if estimate_tokens(body) > available:
            body = truncate_to_sentence(body, available)
            if not body or estimate_tokens(body) < MIN_CHUNK_TOKENS:
                dropped += 1
                continue
            truncated += 1
        cost = overhead + estimate_tokens(body)
        remaining -= cost
        entries.append({
            "tag": tag, "label": label, "body": body, "score": round(chunk.score, 4),
            "chunk_id": chunk.id,
            "document_id": chunk.document.id if chunk.document else None,
            "source_uri": chunk.document.source_uri if chunk.document else "",
        })

    if not entries and not facts_text:
        return ContextBlock(text=NOTHING_RELEVANT, reason=BELOW_THRESHOLD, dropped=dropped)

    # Best material at both ends; the middle is where attention goes to die.
    laid_out = edge_order(entries)
    parts = [HEADER]
    if facts_text:
        parts.append(facts_text)
    for entry in laid_out:
        parts.append(f"{entry['tag']} {entry['label']}\n{entry['body']}")

    text = "\n\n".join(parts)
    block = ContextBlock(
        text=text,
        sources=[{k: entry[k] for k in ("tag", "label", "source_uri", "score", "chunk_id")}
                 for entry in entries],
        tokens=estimate_tokens(text),
        reason=result.reason,
        dropped=dropped,
        truncated=truncated,
    )
    logger.debug("Context assembled: %s", block.describe())
    return block


def citation_map(block: ContextBlock) -> dict[str, str]:
    """{"[S1]": "Wikipedia - Tungsten"} for turning tags back into names."""
    return {entry["tag"]: entry["label"] for entry in block.sources}
