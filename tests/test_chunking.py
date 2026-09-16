"""Phase 2 of the retrieval spec: structure-aware chunking.

No microphone, no network, no model downloads. The tree-sitter tests skip
themselves when the optional grammar pack isn't installed - the line-based
fallback is always exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ev_assistant.chunking import (
    CHARS_PER_TOKEN,
    MAX_TOKENS,
    TARGET_TOKENS,
    TextChunk,
    chunk_code,
    chunk_document,
    chunk_prose,
    chunk_rows,
    context_header,
    detect_language,
    estimate_tokens,
    parse_blocks,
    split_sentences,
)


def has_tree_sitter() -> bool:
    try:
        from tree_sitter_language_pack import get_parser  # noqa: F401
    except ImportError:
        return False
    return True


needs_tree_sitter = pytest.mark.skipif(
    not has_tree_sitter(), reason="tree-sitter language pack not installed"
)


SENTENCES = [
    "Boiling water is the most reliable way to make it safe to drink.",
    "Bring it to a rolling boil and hold it there for one full minute.",
    "Above two thousand metres, hold the boil for three minutes instead.",
    "Let it cool covered so nothing falls in while it sits.",
    "Cloudy water should be filtered through cloth before it is boiled.",
]


def long_prose(repeat: int = 40) -> str:
    """A paragraphed document with known, intact sentences."""
    paragraphs = []
    for i in range(repeat):
        paragraphs.append(" ".join(f"{s[:-1]} ({i})." for s in SENTENCES))
    return "\n\n".join(paragraphs)


# -- token estimation ------------------------------------------------------


def test_estimate_tokens_is_proportional_to_length():
    assert estimate_tokens("") == 1
    assert estimate_tokens("x" * (CHARS_PER_TOKEN * 10)) == 10


# -- sentence splitting ----------------------------------------------------


def test_splits_on_terminators():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_keeps_titles_and_abbreviations_together():
    assert split_sentences("Dr. Smith arrived. He was late.") == [
        "Dr. Smith arrived.",
        "He was late.",
    ]
    assert split_sentences("Bring rope, tarp, etc. Then leave.") == [
        "Bring rope, tarp, etc. Then leave.",
    ]


def test_keeps_initials_and_dotted_abbreviations_together():
    assert split_sentences("He met J. R. R. Tolkien. Twice.") == [
        "He met J. R. R. Tolkien.",
        "Twice.",
    ]
    assert split_sentences("Built in the U.S. It shipped late.") == [
        "Built in the U.S. It shipped late."
    ]


def test_a_lowercase_single_letter_still_ends_a_sentence():
    # "Vitamin C." is a sentence end; "J." in a name is not.
    assert split_sentences("Plot the value of x. Then repeat.") == [
        "Plot the value of x.",
        "Then repeat.",
    ]


def test_handles_quotes_and_repeated_terminators():
    assert split_sentences('She said "go." Then she left.') == [
        'She said "go."',
        "Then she left.",
    ]
    assert split_sentences("Really?! Yes.") == ["Really?!", "Yes."]


def test_empty_and_unterminated_text():
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []
    assert split_sentences("no terminator here") == ["no terminator here"]


# -- block parsing ---------------------------------------------------------


def test_markdown_headings_build_a_path():
    blocks = parse_blocks("# Top\nintro\n\n## Middle\nbody\n\n### Deep\nleaf\n")
    assert [(b.heading_path, b.text) for b in blocks] == [
        ("Top", "intro"),
        ("Top > Middle", "body"),
        ("Top > Middle > Deep", "leaf"),
    ]


def test_wikipedia_headings_are_recognised():
    # What `ev learn <topic>` actually receives: an explaintext extract.
    blocks = parse_blocks("Lead text.\n\n== History ==\nOld stuff.\n\n=== Etymology ===\nName.\n")
    assert [b.heading_path for b in blocks] == ["", "History", "History > Etymology"]


def test_dedenting_a_heading_drops_the_deeper_levels():
    blocks = parse_blocks("# A\n\n## B\nunder b\n\n# C\nunder c\n")
    assert [b.heading_path for b in blocks] == ["A > B", "C"]


def test_setext_heading_with_equals():
    blocks = parse_blocks("Title Here\n=====\nbody text\n")
    assert [(b.heading_path, b.text) for b in blocks] == [("Title Here", "body text")]


def test_paragraphs_split_on_blank_lines():
    blocks = parse_blocks("one one\n\ntwo two\n\n\nthree three")
    assert [b.text for b in blocks] == ["one one", "two two", "three three"]
    assert not any(b.atomic for b in blocks)


def test_lists_tables_and_fences_are_atomic():
    doc = (
        "intro\n\n"
        "- first item\n- second item\n- third item\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "```python\nprint('hi')\n```\n\n"
        "outro\n"
    )
    blocks = parse_blocks(doc)
    kinds = [(b.atomic, b.text.splitlines()[0]) for b in blocks]
    assert kinds[0] == (False, "intro")
    assert kinds[1] == (True, "- first item")
    assert kinds[2] == (True, "| a | b |")
    assert kinds[3] == (True, "```python")
    assert kinds[4] == (False, "outro")


def test_fenced_code_keeps_its_blank_lines():
    blocks = parse_blocks("```\nline one\n\nline two\n```\n")
    assert len(blocks) == 1
    assert blocks[0].atomic
    assert "line one\n\nline two" in blocks[0].text


def test_empty_document_has_no_blocks():
    assert parse_blocks("") == []
    assert parse_blocks("\n\n   \n") == []


# -- context header --------------------------------------------------------


def test_context_header_combinations():
    assert context_header("Tungsten", "History > Etymology") == "Tungsten - History > Etymology"
    assert context_header("Tungsten", "") == "Tungsten"
    assert context_header("", "History") == "History"
    assert context_header("", "") == ""


def test_every_chunk_carries_its_context_header():
    chunks = chunk_prose("== History ==\n" + long_prose(6), title="Tungsten")
    assert chunks
    for chunk in chunks:
        assert chunk.text.startswith("Tungsten - History")
        assert chunk.body in chunk.text
        assert not chunk.body.startswith("Tungsten - History")


def test_a_document_with_no_title_or_headings_has_no_header():
    chunks = chunk_prose("Just some text about nothing in particular.")
    assert len(chunks) == 1
    assert chunks[0].text == chunks[0].body == "Just some text about nothing in particular."


# -- prose chunking --------------------------------------------------------


def test_short_document_is_a_single_chunk():
    chunks = chunk_prose("One short paragraph.", title="T")
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].heading_path == ""


def test_empty_document_produces_no_chunks():
    assert chunk_prose("", title="T") == []
    assert chunk_prose("   \n\n  ", title="T") == []


def test_ordinals_are_sequential_from_zero():
    chunks = chunk_prose(long_prose(), title="Water")
    assert len(chunks) > 3
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunks_stay_within_the_token_ceiling():
    chunks = chunk_prose(long_prose(), title="Water")
    assert max(c.token_count for c in chunks) <= MAX_TOKENS
    # And they aren't all tiny either - the budget is being used.
    assert sum(c.token_count for c in chunks) / len(chunks) > TARGET_TOKENS / 2


def test_the_ceiling_holds_for_a_custom_target():
    chunks = chunk_prose(long_prose(), title="Water", target_tokens=120)
    assert max(c.token_count for c in chunks) <= int(120 * 1.2)
    assert len(chunks) > 6


def test_no_sentence_is_ever_cut_in_half():
    text = long_prose()
    chunks = chunk_prose(text, title="Water")
    joined = "\n".join(c.body for c in chunks)
    for sentence in split_sentences(text.replace("\n\n", " ")):
        assert sentence in joined, f"sentence was split: {sentence!r}"


def test_all_source_text_survives_chunking():
    text = long_prose(8)
    chunks = chunk_prose(text, title="Water")
    joined = " ".join(c.body for c in chunks)
    for paragraph in text.split("\n\n"):
        assert paragraph in joined


def test_consecutive_chunks_overlap_by_whole_sentences():
    chunks = chunk_prose(long_prose(), title="Water", overlap_ratio=0.15)
    assert len(chunks) > 2
    overlapping = 0
    for previous, current in zip(chunks, chunks[1:]):
        head = current.body.split("\n\n")[0]
        if head and head in previous.body:
            overlapping += 1
            # Whole sentences only, never a fragment.
            assert split_sentences(head)[0] in split_sentences(previous.body)
    assert overlapping >= len(chunks) - 2


def test_overlap_can_be_switched_off():
    chunks = chunk_prose(long_prose(), title="Water", overlap_ratio=0.0)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.body.split("\n\n")[0] not in previous.body


def test_heading_path_follows_the_content():
    doc = (
        "== Purification ==\n" + long_prose(10) + "\n\n"
        "== Shelter ==\n" + long_prose(10) + "\n"
    )
    chunks = chunk_prose(doc, title="Survival")
    paths = {c.heading_path for c in chunks}
    assert paths == {"Purification", "Shelter"}
    # Each chunk's own text sits under the heading it claims.
    for chunk in chunks:
        assert chunk.text.startswith(f"Survival - {chunk.heading_path}")


def test_a_stub_section_rides_along_instead_of_becoming_its_own_chunk():
    doc = "== Stub ==\nOne line.\n\n== Real ==\n" + long_prose(6)
    chunks = chunk_prose(doc, title="T")
    assert "One line." in chunks[0].body
    assert chunks[0].heading_path == "Real"  # labelled by what it's mostly made of


def test_a_table_is_never_split_even_when_oversized():
    rows = "\n".join(f"| row {i} | value {i} | note {i} |" for i in range(200))
    doc = f"== Data ==\nlead in\n\n| a | b | c |\n|---|---|---|\n{rows}\n\nafter the table\n"
    chunks = chunk_prose(doc, title="T")
    holding = [c for c in chunks if "| row 0 |" in c.body]
    assert len(holding) == 1
    for i in range(200):
        assert f"| row {i} |" in holding[0].body


def test_a_list_is_never_split_even_when_oversized():
    items = "\n".join(f"- item number {i} with some explanatory text" for i in range(200))
    chunks = chunk_prose(f"== Kit ==\n{items}\n", title="T")
    holding = [c for c in chunks if "- item number 0 " in c.body]
    assert len(holding) == 1
    for i in range(200):
        assert f"- item number {i} " in holding[0].body


def test_a_single_runaway_sentence_is_wrapped_at_word_boundaries():
    # Stripped HTML often arrives with the punctuation gone.
    runaway = " ".join(f"word{i}" for i in range(4000))
    chunks = chunk_prose(runaway, title="T")
    assert len(chunks) > 1
    assert max(c.token_count for c in chunks) <= MAX_TOKENS
    rejoined = " ".join(c.body for c in chunks)
    for i in range(4000):
        assert f"word{i}" in rejoined  # no word was cut in half


# -- code chunking ---------------------------------------------------------


def sample_module(functions: int = 4, body_lines: int = 20) -> str:
    body = "\n".join(f"    value_{i} = compute({i})" for i in range(body_lines))
    out = '"""Module docstring."""\n\nimport os\nimport sys\n\n\n'
    for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")[:functions]:
        out += f"@decorated\ndef {name}(x):\n    \"\"\"Doc for {name}.\"\"\"\n{body}\n    return x\n\n\n"
    out += "class Omega:\n    def method(self):\n        return 1\n\n\nTRAILING = 42\n"
    return out


def test_detect_language_from_suffix():
    assert detect_language("/home/me/thing.py") == "python"
    assert detect_language("https://example.org/a/b.rs?raw=1") == "rust"
    assert detect_language("notes.md") is None
    assert detect_language("") is None


def test_chunk_document_routes_code_by_suffix():
    chunks = chunk_document(sample_module(), title="thing.py", source_uri="/tmp/thing.py")
    assert chunks
    assert "alpha" in chunks[0].heading_path


def test_chunk_document_routes_code_by_source_type():
    chunks = chunk_document(sample_module(), title="x", source_type="code")
    assert chunks
    assert "alpha" in chunks[0].heading_path


def test_code_chunks_never_split_a_definition():
    for language in (None, "python") if has_tree_sitter() else (None,):
        chunks = chunk_code(sample_module(functions=6), title="thing.py", language=language)
        assert len(chunks) > 1
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"):
            holding = [c for c in chunks if f"def {name}(x):" in c.body]
            assert len(holding) == 1, f"{name} appears in {len(holding)} chunks ({language})"
            assert f'"""Doc for {name}."""' in holding[0].body
            # The whole body came along, decorator included.
            assert "@decorated\ndef " + name in holding[0].body
            assert holding[0].body.rstrip().endswith("return x") or "class Omega" in holding[0].body


def test_code_chunk_headings_name_the_definitions():
    chunks = chunk_code(sample_module(), title="thing.py", language=None)
    named = " ".join(c.heading_path for c in chunks)
    for name in ("alpha", "beta", "gamma", "delta", "Omega"):
        assert name in named


def test_module_preamble_is_kept():
    chunks = chunk_code(sample_module(), title="thing.py", language=None)
    assert '"""Module docstring."""' in chunks[0].body
    assert "import os" in chunks[0].body


def test_trailing_code_after_the_last_definition_is_kept():
    chunks = chunk_code(sample_module(), title="thing.py", language=None)
    assert any("TRAILING = 42" in c.body for c in chunks)


def test_code_without_definitions_falls_back_to_line_windows():
    lines = "\n".join(f"value_{i} = {i}" for i in range(500))
    chunks = chunk_code(lines, title="data.py", language=None)
    assert len(chunks) > 1
    assert all(c.heading_path.startswith("L") for c in chunks)
    rejoined = "\n".join(c.body for c in chunks)
    for i in range(500):
        assert f"value_{i} = {i}" in rejoined


def test_empty_code_produces_no_chunks():
    assert chunk_code("", title="t.py") == []
    assert chunk_code("\n\n   ", title="t.py") == []


@needs_tree_sitter
def test_tree_sitter_and_fallback_agree_on_definition_boundaries():
    source = sample_module(functions=6)
    parsed = chunk_code(source, title="thing.py", language="python")
    fallback = chunk_code(source, title="thing.py", language=None)
    assert [c.heading_path for c in parsed] == [c.heading_path for c in fallback]


@needs_tree_sitter
def test_tree_sitter_handles_a_brace_language():
    source = "\n\n".join(
        f"function thing{i}(a, b) {{\n"
        + "\n".join(f"  const someLongVariableName{j} = computeTheThing({j});" for j in range(20))
        + "\n  return a + b;\n}"
        for i in range(12)
    )
    chunks = chunk_code(source, title="app.js", language="javascript")
    assert len(chunks) > 1
    for i in range(12):
        holding = [c for c in chunks if f"function thing{i}(a, b)" in c.body]
        assert len(holding) == 1
        # The whole function came along, not a fragment of it.
        assert "const someLongVariableName0 =" in holding[0].body
        assert "const someLongVariableName19 =" in holding[0].body


@needs_tree_sitter
def test_tree_sitter_finds_definitions_the_regex_fallback_cannot():
    # C function definitions have no keyword to match on, so this only works
    # if the parser is genuinely being used.
    source = "\n\n".join(
        f"int compute_thing_{i}(int a, int b) {{\n"
        + "\n".join(f"  int value_{j} = a * {j} + b;" for j in range(20))
        + "\n  return a + b;\n}"
        for i in range(12)
    )
    parsed = chunk_code(source, title="thing.c", language="c")
    fallback = chunk_code(source, title="thing.c", language=None)

    assert any("compute_thing_0" in c.heading_path for c in parsed)
    assert all(c.heading_path.startswith("L") for c in fallback)
    for i in range(12):
        holding = [c for c in parsed if f"int compute_thing_{i}(" in c.body]
        assert len(holding) == 1
        assert "int value_19 =" in holding[0].body


def test_unparseable_source_still_chunks():
    # A syntax error must not lose the file - tree-sitter is error-tolerant,
    # and the fallback doesn't parse at all.
    broken = "def alpha(:\n    this is not python ((\n" + "\n".join(
        f"line_{i}" for i in range(400)
    )
    chunks = chunk_code(broken, title="broken.py", language="python")
    assert chunks
    assert "line_399" in "\n".join(c.body for c in chunks)


# -- store hand-off --------------------------------------------------------


def test_chunk_rows_match_what_the_store_expects():
    rows = chunk_rows(long_prose(6), title="Water")
    assert rows
    for i, row in enumerate(rows):
        assert set(row) == {"ordinal", "text", "heading_path", "token_count"}
        assert row["ordinal"] == i
        assert isinstance(row["text"], str) and row["text"]
        assert isinstance(row["token_count"], int)


def test_chunks_round_trip_through_the_store(store):
    text = "== Purification ==\n" + long_prose(10)
    rows = chunk_rows(text, title="Water", source_uri="wikipedia:Water")

    doc_id = store.add_document(
        source_uri="wikipedia:Water", source_type="wikipedia", title="Water",
        text=text, chunks=rows,
    )
    assert doc_id is not None

    stored = store.pending_chunks(limit=1000)
    assert len(stored) == len(rows)
    assert [c.ordinal for c in stored] == [r["ordinal"] for r in rows]
    assert all(c.heading_path == "Purification" for c in stored)

    # The context header is in the keyword index too, so a heading-only query
    # still finds the chunk.
    with store._connect() as conn:
        hits = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'purification'"
        ).fetchall()
    assert len(hits) == len(rows)


def test_chunking_a_real_document_loses_nothing(tmp_path):
    # The repo's own README: real markdown, real headings, real lists and
    # code fences, rather than a fixture shaped to suit the parser.
    readme = Path(__file__).resolve().parent.parent / "README.md"
    if not readme.is_file():
        pytest.skip("README.md not present")
    text = readme.read_text(encoding="utf-8")

    chunks = chunk_document(text, title="README", source_uri="README.md")

    assert len(chunks) > 5
    # The ceiling holds, except where an atomic block (a fence, table or list)
    # is itself oversized - those are never split, by design.
    for chunk in chunks:
        if chunk.token_count > MAX_TOKENS:
            assert "```" in chunk.body or "\n|" in chunk.body or "\n- " in chunk.body, (
                f"chunk {chunk.ordinal} is {chunk.token_count} tokens of plain prose"
            )
    assert len({c.heading_path for c in chunks}) > 3
    joined = "\n".join(c.body for c in chunks)
    for line in text.splitlines():
        stripped = line.strip()
        # Heading lines are lifted into heading_path rather than kept inline.
        if not stripped or stripped.startswith("#") or set(stripped) <= set("=-"):
            continue
        assert line in joined, f"lost a line: {line[:60]!r}"


def test_text_chunk_as_row_is_the_dataclass_contract():
    chunk = TextChunk(ordinal=3, text="stored", body="stored", heading_path="A > B", token_count=9)
    assert chunk.as_row() == {
        "ordinal": 3, "text": "stored", "heading_path": "A > B", "token_count": 9,
    }
