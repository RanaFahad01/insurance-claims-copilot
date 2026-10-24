import pytest

from app.rag.chunker import (
    DEFAULT_POLICY_DIR,
    PolicyParseError,
    check_language_parity,
    load_policy_chunks,
    parse_policy,
    policy_hash,
)

GOOD = """# Title of the policy

A disclaimer that is not a clause.

## C1.1 | First clause: covered
Line one of the first clause
continues on a second line.

A second paragraph.

## C1.2 | Second clause
Short text.
"""


def test_real_policy_files_parse_and_match_across_languages():
    chunks = load_policy_chunks()
    en = [c for c in chunks if c.language == "en"]
    es = [c for c in chunks if c.language == "es"]
    assert len(en) == len(es) == 29
    assert [c.clause_id for c in en] == [c.clause_id for c in es]
    assert len({c.chunk_id for c in chunks}) == 58


def test_chunk_ids_combine_language_and_clause():
    chunks = {c.chunk_id: c for c in load_policy_chunks()}
    assert "en-C2.1" in chunks and "es-C2.1" in chunks
    c = chunks["en-C2.1"]
    assert c.clause_id == "C2.1"
    assert c.section == 2
    assert c.title == "Water damage: covered"
    assert c.language == "en"
    assert c.source_file == "policy_en.md"
    assert "sudden and unforeseen" in c.text


def test_metadata_has_clause_title_language_source_and_section():
    c = parse_policy(GOOD, "en", "policy_en.md")[0]
    assert c.metadata() == {
        "clause_id": "C1.1",
        "title": "First clause: covered",
        "language": "en",
        "source_file": "policy_en.md",
        "section": 1,
    }


def test_embedded_document_includes_the_heading_title():
    c = parse_policy(GOOD, "en", "policy_en.md")[0]
    assert c.document.startswith("First clause: covered")
    assert "continues on a second line." in c.document


def test_preamble_before_first_clause_is_ignored():
    chunks = parse_policy(GOOD, "en", "policy_en.md")
    assert [c.clause_id for c in chunks] == ["C1.1", "C1.2"]
    assert "disclaimer" not in " ".join(c.text for c in chunks)


def test_wrapped_lines_join_and_paragraphs_are_kept():
    c = parse_policy(GOOD, "en", "policy_en.md")[0]
    assert c.text == "Line one of the first clause continues on a second line.\n\nA second paragraph."


def test_level_three_heading_is_body_text_not_a_clause():
    text = "## C1.1 | A\nBody.\n### Note\nMore body.\n"
    chunks = parse_policy(text, "en", "policy_en.md")
    assert len(chunks) == 1
    assert "### Note" in chunks[0].text


def test_malformed_heading_fails_loudly_with_line_number():
    text = "## C1.1 | A\nBody.\n## C1.2 Missing the pipe\nBody.\n"
    with pytest.raises(PolicyParseError, match=r"policy_en\.md:3"):
        parse_policy(text, "en", "policy_en.md")


def test_non_clause_level_two_heading_fails():
    with pytest.raises(PolicyParseError, match="not in the form"):
        parse_policy("## Introduction\nText.\n## C1.1 | A\nBody.\n", "en", "policy_en.md")


def test_duplicate_clause_id_fails():
    text = "## C1.1 | A\nBody.\n## C1.1 | B\nBody.\n"
    with pytest.raises(PolicyParseError, match="duplicate clause C1.1"):
        parse_policy(text, "en", "policy_en.md")


def test_empty_clause_fails():
    with pytest.raises(PolicyParseError, match="C1.1 has no text"):
        parse_policy("## C1.1 | A\n\n## C1.2 | B\nBody.\n", "en", "policy_en.md")
    with pytest.raises(PolicyParseError, match="C1.2 has no text"):
        parse_policy("## C1.1 | A\nBody.\n## C1.2 | B\n", "en", "policy_en.md")


def test_file_without_clauses_fails():
    with pytest.raises(PolicyParseError, match="no clauses found"):
        parse_policy("# Only a title\n\nNo clauses here.\n", "en", "policy_en.md")


def test_language_parity_mismatch_fails():
    en = parse_policy("## C1.1 | A\nBody.\n## C1.2 | B\nBody.\n", "en", "policy_en.md")
    es = parse_policy("## C1.1 | A\nTexto.\n", "es", "policy_es.md")
    with pytest.raises(PolicyParseError, match=r"only in en: \['C1.2'\]"):
        check_language_parity(en + es)


def test_bad_policy_filename_fails(tmp_path):
    (tmp_path / "policy_english.md").write_text(GOOD, encoding="utf-8")
    with pytest.raises(PolicyParseError, match="policy_<two-letter language>.md"):
        load_policy_chunks(tmp_path)


def test_empty_policy_folder_fails(tmp_path):
    with pytest.raises(PolicyParseError, match="no policy_"):
        load_policy_chunks(tmp_path)


def test_policy_hash_is_stable_and_ignores_line_endings(tmp_path):
    lf = tmp_path / "lf"
    crlf = tmp_path / "crlf"
    lf.mkdir()
    crlf.mkdir()
    (lf / "policy_en.md").write_bytes(GOOD.encode("utf-8"))
    (crlf / "policy_en.md").write_bytes(GOOD.replace("\n", "\r\n").encode("utf-8"))
    assert policy_hash(lf) == policy_hash(crlf)
    assert len(policy_hash(lf)) == 64
    assert policy_hash(DEFAULT_POLICY_DIR) == policy_hash(DEFAULT_POLICY_DIR)


def test_policy_hash_changes_when_content_changes(tmp_path):
    (tmp_path / "policy_en.md").write_text(GOOD, encoding="utf-8")
    before = policy_hash(tmp_path)
    (tmp_path / "policy_en.md").write_text(GOOD.replace("Short text.", "Changed text."), encoding="utf-8")
    assert policy_hash(tmp_path) != before
