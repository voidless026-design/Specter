"""Query parsing helpers for the reporting tool."""


def parse_query(text):
    """Split a raw query string into normalised terms."""
    if not text:
        return []
    return [term.strip().lower() for term in text.split() if term.strip()]


def build_filter(terms, field="body"):
    """Turn parsed terms into a SQL LIKE filter and its parameters."""
    if not terms:
        return "", []
    clause = " AND ".join(f"{field} LIKE ?" for _ in terms)
    return clause, [f"%{term}%" for term in terms]


class QueryError(ValueError):
    """Raised when a query cannot be parsed at all."""


def validate(terms, maximum=32):
    if len(terms) > maximum:
        raise QueryError(f"too many terms: {len(terms)} > {maximum}")
    return terms
