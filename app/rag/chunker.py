"""Split the policy files into one chunk per clause. Pure code: no network, no embeddings."""
import hashlib
import re
from pathlib import Path

from pydantic import BaseModel

DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"

CLAUSE_HEADING = re.compile(r"^## (C(\d+)\.(\d+)) \| (\S.*?)\s*$")
POLICY_FILENAME = re.compile(r"^policy_([a-z]{2})\.md$")


class PolicyParseError(ValueError):
    """The policy files are malformed. Raised loudly so a bad file never reaches the index."""


class PolicyChunk(BaseModel):
    chunk_id: str      # language + clause, e.g. "en-C4.1", so versions never collide
    clause_id: str     # e.g. "C4.1", the same in every language
    section: int       # the number before the dot
    title: str
    text: str          # the clause body, shown to people and to the model
    language: str
    source_file: str

    @property
    def document(self) -> str:
        """The text that gets embedded: the heading title followed by the body."""
        return f"{self.title}\n\n{self.text}"

    def metadata(self) -> dict[str, str | int]:
        return {
            "clause_id": self.clause_id,
            "title": self.title,
            "language": self.language,
            "source_file": self.source_file,
            "section": self.section,
        }


def _join_body(lines: list[str]) -> str:
    """Join wrapped lines into paragraphs. A blank line starts a new paragraph."""
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        stripped = line.strip()
        if stripped:
            paragraphs[-1].append(stripped)
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join(" ".join(p) for p in paragraphs if p)


def parse_policy(text: str, language: str, source_file: str) -> list[PolicyChunk]:
    """Parse one policy file. Text before the first clause heading (title, disclaimer) is ignored."""
    chunks: list[PolicyChunk] = []
    seen: set[str] = set()
    current: dict | None = None

    def finish() -> None:
        if current is None:
            return
        body = _join_body(current["lines"])
        if not body:
            raise PolicyParseError(f"{source_file}: clause {current['clause_id']} has no text")
        chunks.append(PolicyChunk(
            chunk_id=f"{language}-{current['clause_id']}",
            clause_id=current["clause_id"],
            section=current["section"],
            title=current["title"],
            text=body,
            language=language,
            source_file=source_file,
        ))

    for number, line in enumerate(text.splitlines(), start=1):
        match = CLAUSE_HEADING.match(line)
        if match:
            finish()
            clause_id = match.group(1)
            if clause_id in seen:
                raise PolicyParseError(f"{source_file}:{number}: duplicate clause {clause_id}")
            seen.add(clause_id)
            current = {"clause_id": clause_id, "section": int(match.group(2)), "title": match.group(4), "lines": []}
        elif line.startswith("## "):
            raise PolicyParseError(f"{source_file}:{number}: heading is not in the form '## C1.1 | Title': {line!r}")
        elif current is not None:
            current["lines"].append(line)
    finish()

    if not chunks:
        raise PolicyParseError(f"{source_file}: no clauses found")
    return chunks


def check_language_parity(chunks: list[PolicyChunk]) -> None:
    """Every language must contain exactly the same clause IDs."""
    by_language: dict[str, set[str]] = {}
    for chunk in chunks:
        by_language.setdefault(chunk.language, set()).add(chunk.clause_id)
    reference_language = sorted(by_language)[0]
    reference = by_language[reference_language]
    for language, ids in by_language.items():
        if ids != reference:
            raise PolicyParseError(
                f"clause IDs differ between '{reference_language}' and '{language}': "
                f"only in {reference_language}: {sorted(reference - ids)}, only in {language}: {sorted(ids - reference)}"
            )


def _policy_files(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.glob("policy_*.md"))
    if not files:
        raise PolicyParseError(f"no policy_*.md files in {directory}")
    for path in files:
        if not POLICY_FILENAME.match(path.name):
            raise PolicyParseError(f"{path.name}: policy files must be named policy_<two-letter language>.md")
    return files


def load_policy_chunks(directory: Path = DEFAULT_POLICY_DIR) -> list[PolicyChunk]:
    """Parse every policy_<lang>.md in the folder and check the languages match."""
    chunks: list[PolicyChunk] = []
    for path in _policy_files(directory):
        language = POLICY_FILENAME.match(path.name).group(1)
        chunks.extend(parse_policy(path.read_text(encoding="utf-8"), language, path.name))
    check_language_parity(chunks)
    return chunks


def policy_hash(directory: Path = DEFAULT_POLICY_DIR) -> str:
    """SHA-256 of the policy files' content, so every result can say which policy version it used.

    Line endings are normalised, so the hash is the same on Windows and elsewhere.
    """
    digest = hashlib.sha256()
    for path in _policy_files(directory):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()
