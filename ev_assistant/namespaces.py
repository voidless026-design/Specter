"""Namespaces - what stops a coding question from pulling chemistry chunks.

Every document gets a namespace at ingest time, and every query gets a
*soft* preference at retrieval time. Soft matters: a hard filter is one
misrouted query away from E.V. confidently knowing nothing, so by default
the routed namespace is boosted and the others are demoted, not excluded.
Only a caller that explicitly asks gets a hard filter.

``personal`` is the exception to everything. It is always searched and
always boosted above the rest, because what E.V. knows about you outranks
what she read on the internet.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PERSONAL = "personal"
CODE = "code"
REFERENCE = "reference"
NEWS = "news"
DOMAIN_PREFIX = "domain:"

BUILTIN = (PERSONAL, CODE, REFERENCE, NEWS)
DEFAULT_NAMESPACE = REFERENCE

# Relative weights applied to a candidate's fused score. The routed namespace
# is the reference point; everything else is demoted rather than dropped.
ROUTED_WEIGHT = 1.0
PERSONAL_WEIGHT = 1.25
OTHER_WEIGHT = 0.6

_VALID_RE = re.compile(r"^(?:[a-z][a-z0-9_-]*|domain:[a-z0-9][a-z0-9_-]*)$")

# Sensible defaults by source_type, overridable per source in config.
SOURCE_TYPE_ROUTES = {
    "conversation": PERSONAL,
    "voice": PERSONAL,
    "transcript": PERSONAL,
    "note": PERSONAL,
    "file": PERSONAL,        # a file you pointed her at is yours, not reference
    "code": CODE,
    "feed": NEWS,
    "rss": NEWS,
    "news": NEWS,
    "wikipedia": REFERENCE,
    "web": REFERENCE,
    "pdf": REFERENCE,
    "epub": REFERENCE,
}

# Source suffixes that mean "this is source code" regardless of source_type.
_CODE_SUFFIXES = (
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".go", ".rs",
    ".rb", ".java", ".c", ".h", ".cpp", ".cc", ".hpp", ".cs", ".php", ".swift",
    ".kt", ".scala", ".lua", ".sh", ".bash", ".sql",
)
_CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org")

# Phrases in a question that name where to look. Phase 5 calls detect();
# the vocabulary lives here so namespace knowledge stays in one place.
HINT_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    (PERSONAL, re.compile(
        r"\b(my|our) (notes?|files?|documents?|calendar|preferences?|settings?)\b"
        r"|\bwhat (did|have) (i|we)\b|\bi told you\b|\bwe (discussed|talked about)\b"
        r"|\babout me\b|\bmy (name|birthday|address|job|laptop|machine|pc)\b", re.I)),
    (CODE, re.compile(
        r"\b(in|from) (the |my )?(code ?base|repo(sitory)?|source|project)\b"
        r"|\b(function|method|class|module|variable|traceback|stack ?trace)\b"
        r"|\b(compil|debug|refactor)\w*\b", re.I)),
    (NEWS, re.compile(
        r"\b(latest|recent|today'?s?|yesterday'?s?|this (week|morning)|breaking)\b"
        r"|\b(news|headlines?|current events)\b", re.I)),
    (REFERENCE, re.compile(
        r"\b(wikipedia|encyclopa?edia)\b|\bwho was\b|\bwhat is the (history|origin)\b", re.I)),
)


def normalize(namespace: str | None) -> str:
    """Canonical form, or the default when it's blank or malformed."""
    value = (namespace or "").strip().lower()
    if not value:
        return DEFAULT_NAMESPACE
    value = value.replace(" ", "-")
    if not _VALID_RE.match(value):
        logger.warning("Ignoring malformed namespace %r, using %s", namespace, DEFAULT_NAMESPACE)
        return DEFAULT_NAMESPACE
    return value


def is_valid(namespace: str | None) -> bool:
    return bool(_VALID_RE.match((namespace or "").strip().lower()))


def domain(name: str) -> str:
    """Build a user-defined namespace, e.g. domain("Organic Chem")."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    if not slug:
        raise ValueError("A domain namespace needs a name.")
    return f"{DOMAIN_PREFIX}{slug}"


def looks_like_code(source_uri: str) -> bool:
    uri = (source_uri or "").split("?")[0].split("#")[0].lower()
    if uri.endswith(_CODE_SUFFIXES):
        return True
    return any(host in uri for host in _CODE_HOSTS)


def route(source_type: str, source_uri: str = "", rules: dict | None = None) -> str:
    """Pick the namespace for a document being ingested.

    Order: an explicit config rule, then the file/URL shape, then the
    source_type default, then `reference`.
    """
    for pattern, namespace in (rules or {}).items():
        if _matches(pattern, source_uri, source_type):
            return normalize(namespace)
    if looks_like_code(source_uri):
        return CODE
    return SOURCE_TYPE_ROUTES.get((source_type or "").strip().lower(), DEFAULT_NAMESPACE)


def _matches(pattern: str, source_uri: str, source_type: str) -> bool:
    """Config rules match a substring of the URI, or `type:<source_type>`."""
    pattern = (pattern or "").strip().lower()
    if not pattern:
        return False
    if pattern.startswith("type:"):
        return pattern[5:] == (source_type or "").strip().lower()
    return pattern in (source_uri or "").lower()


def detect(text: str) -> list[str]:
    """Namespaces a question points at, in the order the cues appear.

    Returns [] when nothing in the wording says where to look - which is the
    common case, and why routing is a preference rather than a filter.
    """
    found: list[tuple[int, str]] = []
    for namespace, pattern in HINT_PATTERNS:
        match = pattern.search(text or "")
        if match:
            found.append((match.start(), namespace))
    return [ns for _, ns in sorted(found)]


@dataclass
class NamespacePlan:
    """How a single retrieval should treat each namespace."""

    routed: list[str] = field(default_factory=list)
    restrict: bool = False
    weights: dict[str, float] = field(default_factory=dict)
    other_weight: float = OTHER_WEIGHT
    allowed: list[str] | None = None   # None means "everything"

    def weight(self, namespace: str | None) -> float:
        return self.weights.get(normalize(namespace), self.other_weight)

    def allows(self, namespace: str | None) -> bool:
        if self.allowed is None:
            return True
        return normalize(namespace) in self.allowed

    def sql_filter(self, column: str = "d.namespace") -> tuple[str, list]:
        """A WHERE fragment for the hard-filter case, ('', []) otherwise."""
        if self.allowed is None:
            return "", []
        placeholders = ",".join("?" * len(self.allowed))
        return f"{column} IN ({placeholders})", list(self.allowed)

    def describe(self) -> str:
        mode = "restricted to" if self.restrict else "boosting"
        return f"{mode} {', '.join(self.routed) or 'nothing in particular'}"


def plan(
    routed: Iterable[str] | None = None,
    *,
    restrict: bool = False,
    include_personal: bool = True,
    cfg=None,
) -> NamespacePlan:
    """Build the retrieval-time namespace preference.

    `routed` is what query understanding decided the question is about.
    `restrict` turns the preference into a hard filter - only when a caller
    explicitly asks, because a misrouted query under a hard filter finds
    nothing at all.
    """
    personal_weight = float(getattr(cfg, "personal_namespace_boost", PERSONAL_WEIGHT))
    routed_weight = float(getattr(cfg, "routed_namespace_boost", ROUTED_WEIGHT))
    other_weight = float(getattr(cfg, "other_namespace_weight", OTHER_WEIGHT))

    picked = [normalize(ns) for ns in (routed or []) if is_valid(str(ns))]
    # Preserve order, drop repeats.
    picked = list(dict.fromkeys(picked))

    weights = {ns: routed_weight for ns in picked}
    if include_personal:
        # Always searched, always boosted - even when the question routed
        # somewhere else entirely.
        weights[PERSONAL] = max(personal_weight, weights.get(PERSONAL, 0.0))

    allowed = None
    if restrict and picked:
        allowed = list(dict.fromkeys(picked + ([PERSONAL] if include_personal else [])))

    return NamespacePlan(
        routed=picked, restrict=bool(restrict and picked), weights=weights,
        other_weight=other_weight, allowed=allowed,
    )


def plan_for_query(text: str, *, restrict: bool = False, cfg=None) -> NamespacePlan:
    """Convenience: detect namespace cues in a question and plan from them."""
    return plan(detect(text), restrict=restrict, cfg=cfg)
