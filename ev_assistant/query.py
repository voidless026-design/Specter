"""Query understanding - deciding what, if anything, to look up.

Retrieval is not free and it is not always right, so the first job here is to
decide whether the knowledge base should be consulted at all. "Hey E.V.",
"what's 17 times 3" and "open Firefox" have no business touching the store,
and retrieving on everything is the standard way to make an assistant worse.

Then, for the questions that do need it:

- Conversational questions are rewritten standalone. "What about its melting
  point?" retrieves nothing useful; "What is the melting point of tungsten?"
  retrieves the right paragraph.
- The question is expanded into a few variants, because the three retrieval
  channels want different shapes of text: a natural question for the dense
  index, bare keywords for BM25, and a HyDE paragraph - a hypothetical
  *answer* - which embeds closer to real document prose than a question does.
- Temporal, namespace and entity filters are pulled out of the wording.

Everything has a rules-only path. The cheap model sharpens the decisions when
it's reachable; with no brain at all, E.V. still retrieves sensibly.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from ev_assistant import namespaces
from ev_assistant.namespaces import NamespacePlan

logger = logging.getLogger(__name__)

MAX_VARIANTS = 4
HYDE_MAX_TOKENS = 160
REWRITE_MAX_TOKENS = 80
CLASSIFY_MAX_TOKENS = 8

DAY = 86400.0

# Words with no discriminating power in a keyword query.
_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from get give had has have
he her hers him his how i if in into is it its me my of on or our ours she should so
some tell that the their them then there these they this those to us was we were what
when where which who whom why will with would you your yours about
""".split())

# Unicode-aware on purpose: an ASCII-only class shreds "ünïcödé" into single
# letters, and the embedder (bge-m3) is multilingual.
_WORD_RE = re.compile(r"\w[\w'.-]*")
_QUOTED_RE = re.compile(r'"([^"]{2,60})"')
_PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
_IDENTIFIER_RE = re.compile(r"\b([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*|[a-z]+_[a-z_0-9]+|\w+\(\))")

# Questions that never need the store.
# One or more filler phrases and nothing else - "hey", "thanks", "okay sure".
_FILLER = (
    r"hi|hey|hello|yo|sup|thanks?|thank you|cheers|ta|good (?:morning|evening|night)|"
    r"how are you|how'?s it going|you there|are you (?:there|awake|up)|never ?mind|"
    r"forget it|shut up|stop|cancel|nothing|ok(?:ay)?|cool|nice|lol|goodbye|bye|see ya|"
    r"hm+|uh+|um+|er+|huh|yeah|yep|nope|nah|right|sure|wow|oops|please|mate"
)
_CHITCHAT_RE = re.compile(
    rf"^\s*(?:{_FILLER})(?:[\s,]+(?:{_FILLER}))*\s*[!.?]*$", re.I)
_ARITHMETIC_RE = re.compile(
    r"^\s*(what'?s?|what is|calculate|compute|how much is)?\s*"
    r"[-+(]?\s*\d[\d\s.,]*\s*([-+*/x×÷^]|plus|minus|times|divided by|over)\s*[\d(].*$", re.I)
_TIME_RE = re.compile(
    r"^\s*(?:what'?s?|what is|tell me)?\s*(?:the )?(?:time|date|day)"
    r"(?:\s+(?:is it|today|now|right now))*\s*[?.!]*$", re.I)
_COMMAND_RE = re.compile(
    r"^\s*(open|close|quit|launch|start|run|play|pause|stop|mute|unmute|volume|turn|set|"
    r"increase|decrease|skip|next|previous|shut ?down|reboot|restart|lock|enable|disable|install)\b",
    re.I)

# Questions that definitely do need it.
_STORE_RE = re.compile(
    r"\b(my|our) (notes?|files?|documents?|stuff)\b|\bwhat did (i|we)\b|\bi told you\b|"
    r"\bin the (codebase|repo|repository|documents?|notes?)\b|\baccording to\b|"
    r"\bwhat do you (know|remember) about\b|\blook (this )?up\b|\bsearch (your|the)\b", re.I)

# Conversational fragments that can't stand on their own.
_DANGLING_RE = re.compile(
    r"^\s*(and |but |so |what about|how about|and what of)\b|"
    r"\b(it|its|it's|that|this|those|these|they|them|their|he|she|him|her|his|hers|one)\b", re.I)

_RELATIVE_PERIODS = {
    "today": 1, "yesterday": 2, "this week": 7, "last week": 14, "past week": 7,
    "this month": 31, "last month": 62, "past month": 31, "recently": 30,
    "lately": 30, "this year": 365, "last year": 730, "past year": 365,
}
_SINCE_YEAR_RE = re.compile(r"\b(?:since|after|from)\s+(\d{4})\b", re.I)
_BEFORE_YEAR_RE = re.compile(r"\b(?:before|until|up to|prior to)\s+(\d{4})\b", re.I)
_IN_YEAR_RE = re.compile(r"\bin\s+(\d{4})\b", re.I)
_LAST_N_RE = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s+(day|week|month|year)s?\b", re.I)


@dataclass
class Filters:
    """Constraints pulled out of the question's wording."""

    published_after: float | None = None
    published_before: float | None = None
    entities: list[str] = field(default_factory=list)

    @property
    def temporal(self) -> bool:
        return self.published_after is not None or self.published_before is not None

    def describe(self) -> str:
        parts = []
        if self.published_after:
            parts.append(f"after {_date(self.published_after)}")
        if self.published_before:
            parts.append(f"before {_date(self.published_before)}")
        if self.entities:
            parts.append(f"entities={self.entities}")
        return ", ".join(parts) or "none"


@dataclass
class QueryVariant:
    """One phrasing of the question, aimed at one retrieval channel."""

    text: str
    kind: str        # question | keyword | hyde | context
    channel: str     # dense | sparse | both


@dataclass
class RetrievalPlan:
    """Everything retrieval needs to know about one question."""

    original: str
    question: str
    needs_retrieval: bool
    reason: str
    variants: list[QueryVariant] = field(default_factory=list)
    namespace_plan: NamespacePlan = field(default_factory=namespaces.plan)
    filters: Filters = field(default_factory=Filters)
    rewritten: bool = False
    classified_by: str = "rules"   # rules | model | default

    def texts(self, channel: str | None = None) -> list[str]:
        return [v.text for v in self.variants
                if channel is None or v.channel in (channel, "both")]

    def log(self) -> None:
        if not self.needs_retrieval:
            logger.info("Retrieval skipped (%s, by %s): %r",
                        self.reason, self.classified_by, self.original)
            return
        logger.info(
            "Retrieval plan for %r: question=%r variants=%s namespaces=[%s] filters=[%s] (%s)",
            self.original, self.question,
            [(v.kind, v.text[:60]) for v in self.variants],
            self.namespace_plan.describe(), self.filters.describe(), self.classified_by,
        )


def _date(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(ts))


# ---------------------------------------------------------------------------
# Step 1: does this need retrieval at all?
# ---------------------------------------------------------------------------


def classify_by_rules(text: str) -> tuple[bool, str] | None:
    """(needs_retrieval, reason), or None when the rules aren't confident."""
    stripped = (text or "").strip()
    if not stripped:
        return False, "empty"
    if _STORE_RE.search(stripped):
        return True, "names the store"
    if _CHITCHAT_RE.match(stripped):
        return False, "chitchat"
    if _TIME_RE.match(stripped):
        return False, "clock"
    if _ARITHMETIC_RE.match(stripped):
        return False, "arithmetic"
    if _COMMAND_RE.match(stripped) and "?" not in stripped:
        return False, "device command"
    # Content words, not total words: "bowline knot" and "tungsten melting"
    # are perfectly good searches, while "ok then" and "and so" are not.
    if not [w for w in _WORD_RE.findall(stripped) if w.lower() not in _STOPWORDS]:
        return False, "nothing to search on"
    return None


_CLASSIFY_SYSTEM = (
    "You decide whether a question needs a lookup in a personal knowledge base "
    "of saved documents, notes and articles.\n"
    "Answer SEARCH if answering well needs specific facts, documents, personal "
    "details, recent events, or anything the asker has saved.\n"
    "Answer SKIP if it is small talk, a command to a computer, arithmetic, or "
    "general knowledge any competent assistant already has.\n"
    "Reply with exactly one word: SEARCH or SKIP."
)


def classify(text: str, cfg=None, complete=None) -> tuple[bool, str, str]:
    """(needs_retrieval, reason, decided_by). Rules first, model only if unsure."""
    verdict = classify_by_rules(text)
    if verdict is not None:
        return verdict[0], verdict[1], "rules"
    if cfg is not None and complete is not None:
        answer = complete(cfg, _CLASSIFY_SYSTEM, text.strip(), CLASSIFY_MAX_TOKENS)
        if answer:
            word = answer.strip().upper()
            if word.startswith("SKIP"):
                return False, "model says the brain already knows this", "model"
            if word.startswith("SEARCH"):
                return True, "model says look it up", "model"
            logger.debug("Classifier returned something unexpected: %r", answer)
    # No brain, or it didn't answer. Retrieve: the relevance floor in Phase 7
    # throws away bad matches, so an unnecessary lookup costs time, not quality.
    return True, "unclear, and an empty result is cheap", "default"


# ---------------------------------------------------------------------------
# Step 2: rewrite conversational questions standalone
# ---------------------------------------------------------------------------


_REWRITE_SYSTEM = (
    "Rewrite the user's latest question so it stands on its own, using the "
    "conversation for context. Resolve pronouns and fill in what was left "
    "implicit. Keep it short and keep the original meaning. "
    "Reply with the rewritten question only - no preamble, no quotes."
)


def needs_rewrite(text: str) -> bool:
    """Whether the question leans on something said earlier."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    words = _WORD_RE.findall(stripped)
    if len(words) > 18:
        return False  # long questions generally carry their own context
    return bool(_DANGLING_RE.search(stripped))


def salient_topic(history: list[dict]) -> str:
    """The most recent thing the conversation was about, roughly.

    A proper noun if there is one, otherwise the longest content word. Crude,
    but it only has to be good enough to add a retrieval variant.
    """
    for turn in reversed(history or []):
        content = str(turn.get("content", ""))
        proper = _PROPER_RE.findall(content)
        if proper:
            return max(proper, key=len)
        words = [w for w in _WORD_RE.findall(content)
                 if w.lower() not in _STOPWORDS and len(w) > 3]
        if words:
            return max(words, key=len)
    return ""


def rewrite(text: str, history: list[dict] | None = None, cfg=None, complete=None
            ) -> tuple[str, bool]:
    """(standalone question, was_rewritten)."""
    if not history or not needs_rewrite(text):
        return text.strip(), False
    if cfg is not None and complete is not None:
        transcript = "\n".join(
            f"{t.get('role', 'user')}: {t.get('content', '')}" for t in history[-4:]
        )
        answer = complete(
            cfg, _REWRITE_SYSTEM, f"{transcript}\nuser: {text.strip()}\n\nRewritten question:",
            REWRITE_MAX_TOKENS,
        )
        if answer:
            # First line, then unquote - a chatty model adds a second line, and
            # unquoting first would leave the closing quote stuck on line one.
            first = answer.strip().splitlines()[0].strip()
            cleaned = first.strip('"').strip("'").strip()
            if cleaned and len(cleaned) < 400:
                return cleaned, True
    return text.strip(), False


# ---------------------------------------------------------------------------
# Step 3: query variants
# ---------------------------------------------------------------------------


def keyword_query(text: str) -> str:
    """Content words only - what BM25 actually scores on."""
    words = [w for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS]
    seen, out = set(), []
    for word in words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            out.append(word)
    return " ".join(out)


def fts_match_query(text: str) -> str:
    """A safe FTS5 MATCH expression.

    FTS5 treats quotes, `*`, `:`, `^`, NEAR/AND/OR/NOT as syntax, so a raw
    question is a syntax error waiting to happen. Every term is quoted and
    OR-ed, which is both valid and the recall-friendly reading.
    """
    terms = [w for w in _WORD_RE.findall(text or "") if w.lower() not in _STOPWORDS]
    quoted = [f'"{t.replace(chr(34), "")}"' for t in terms if t.strip('.-_')]
    return " OR ".join(dict.fromkeys(quoted))


_HYDE_SYSTEM = (
    "Write a short, factual paragraph that would answer the user's question, "
    "as if it were an excerpt from a reference document. Two or three "
    "sentences. State it plainly and confidently - it is used to search a "
    "document index, not shown to anyone. No preamble, no hedging."
)


def hyde(text: str, cfg=None, complete=None) -> str:
    """A hypothetical answer paragraph, or "" when no brain is reachable.

    Embedding a plausible answer beats embedding the question: documents are
    written as answers, so an answer-shaped vector lands nearer to them.
    """
    if cfg is None or complete is None:
        return ""
    answer = complete(cfg, _HYDE_SYSTEM, text.strip(), HYDE_MAX_TOKENS)
    return (answer or "").strip()


def build_variants(question: str, *, topic: str = "", cfg=None, complete=None
                   ) -> list[QueryVariant]:
    """2-4 phrasings, each aimed at the channel it suits."""
    variants = [QueryVariant(question, "question", "dense")]

    keywords = keyword_query(question)
    if keywords and keywords.lower() != question.lower():
        variants.append(QueryVariant(keywords, "keyword", "sparse"))
    elif keywords:
        variants[0] = QueryVariant(question, "question", "both")

    # No model for a real rewrite? Pairing the topic with the question is a
    # blunt but genuinely effective stand-in for retrieval purposes.
    if topic and topic.lower() not in question.lower():
        variants.append(QueryVariant(f"{topic} {question}", "context", "both"))

    passage = hyde(question, cfg, complete)
    if passage:
        variants.append(QueryVariant(passage, "hyde", "dense"))

    return variants[:MAX_VARIANTS]


# ---------------------------------------------------------------------------
# Step 4: filters
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[str]:
    """Names, quoted phrases and code identifiers worth matching exactly."""
    found: list[str] = []
    found.extend(m.strip() for m in _QUOTED_RE.findall(text or ""))
    # Skip a sentence-initial capital - it's grammar, not a name.
    for match in _PROPER_RE.finditer(text or ""):
        if match.start() > 0:
            found.append(match.group(1))
    found.extend(_IDENTIFIER_RE.findall(text or ""))
    seen, out = set(), []
    for item in found:
        key = item.lower()
        if key and key not in seen and key not in _STOPWORDS:
            seen.add(key)
            out.append(item)
    return out[:8]


def extract_temporal(text: str, now: float | None = None) -> tuple[float | None, float | None]:
    """(published_after, published_before) from phrases like 'since 2023'."""
    now = time.time() if now is None else now
    lowered = (text or "").lower()
    after = before = None

    match = _LAST_N_RE.search(lowered)
    if match:
        span = {"day": 1, "week": 7, "month": 31, "year": 365}[match.group(2)]
        after = now - int(match.group(1)) * span * DAY
    else:
        for phrase, days in _RELATIVE_PERIODS.items():
            if phrase in lowered:
                after = now - days * DAY
                break

    match = _SINCE_YEAR_RE.search(lowered)
    if match:
        after = _year_start(int(match.group(1)))
    match = _BEFORE_YEAR_RE.search(lowered)
    if match:
        before = _year_start(int(match.group(1)))
    match = _IN_YEAR_RE.search(lowered)
    if match and after is None and before is None:
        year = int(match.group(1))
        after, before = _year_start(year), _year_start(year + 1)

    return after, before


def _year_start(year: int) -> float:
    import calendar
    return float(calendar.timegm((year, 1, 1, 0, 0, 0, 0, 1, 0)))


def extract_filters(text: str, now: float | None = None) -> Filters:
    after, before = extract_temporal(text, now)
    return Filters(published_after=after, published_before=before,
                   entities=extract_entities(text))


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------


def build_plan(
    text: str,
    history: list[dict] | None = None,
    cfg=None,
    complete=None,
    *,
    restrict_namespaces: bool = False,
    now: float | None = None,
) -> RetrievalPlan:
    """Turn a raw question into a RetrievalPlan, and log it.

    `complete` is the utility-completion callable (providers.complete). Pass
    None to stay on the rules-only path - which is what the tests, and any
    machine with no brain reachable, do.
    """
    original = (text or "").strip()
    needed, reason, decided_by = classify(original, cfg, complete)
    if not needed:
        plan = RetrievalPlan(
            original=original, question=original, needs_retrieval=False,
            reason=reason, classified_by=decided_by,
        )
        plan.log()
        return plan

    question, rewritten = rewrite(original, history, cfg, complete)
    topic = "" if rewritten else salient_topic(history or [])
    plan = RetrievalPlan(
        original=original,
        question=question,
        needs_retrieval=True,
        reason=reason,
        variants=build_variants(question, topic=topic, cfg=cfg, complete=complete),
        namespace_plan=namespaces.plan(
            namespaces.detect(f"{original} {question}"),
            restrict=restrict_namespaces, cfg=cfg,
        ),
        filters=extract_filters(question, now),
        rewritten=rewritten,
        classified_by=decided_by,
    )
    plan.log()
    return plan
