"""docs/codes.md must list exactly the codes the code can produce. No API calls.

Most sets are read from the code (Literal types, the reason strings in the rules, the error strings of the tools), so adding a
code without documenting it, or documenting one that does not exist, fails here.
"""
import re
from pathlib import Path
from typing import get_args

from app.agent.documents import DocumentError  # noqa: F401  (kinds are read from the source below)
from app.agent.reviews import Resolution, ReviewStatus
from app.agent.state import AgentStatus, Mismatch
from app.rag.coverage import CoverageStatus, GateName
from app.rag.dossier import GATE_REASONS, FinalRoute
from app.rules import REQUIRED_FIELDS

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "codes.md").read_text(encoding="utf-8")
SOURCE = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "app").rglob("*.py")}


def documented(heading: str) -> list[str]:
    """The codes in the first column of the tables under '## heading'."""
    section = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", DOC, re.S | re.M)
    assert section, f"no section '{heading}' in docs/codes.md"
    return re.findall(r"^\| `([^`]+)` \|", section.group(1), re.M)


def source_strings(filename: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, SOURCE[filename]))


def test_every_documented_code_is_listed_once():
    for heading in re.findall(r"^## (.+)$", DOC, re.M):
        codes = documented(heading)
        assert codes and len(codes) == len(set(codes)), heading


def test_routes():
    assert set(documented("Routes")) == set(get_args(FinalRoute))


def test_reasons_are_exactly_the_ones_the_rules_the_policy_check_and_the_agent_can_add():
    code = {f"missing_{field}" for field in REQUIRED_FIELDS}
    code |= source_strings("rules.py", r'reasons\.append\("([a-z_]+)"\)')
    code |= set(GATE_REASONS.values())
    code |= source_strings("dossier.py", r'reasons\.append\("([a-z_]+)"\)')
    code |= source_strings("finalize.py", r'reasons\.append\("([a-z_]+)"\)')
    assert len(code) == 16
    assert set(documented("Reasons")) == code


def test_coverage_codes():
    assert set(documented("Coverage status")) == set(get_args(CoverageStatus))
    assert set(documented("Coverage gate")) == set(get_args(GateName))


def test_mismatch_codes():
    assert set(documented("Mismatch fields")) == set(get_args(Mismatch.model_fields["field"].annotation))
    assert set(documented("Mismatch severity")) == set(get_args(Mismatch.model_fields["severity"].annotation))


def test_agent_codes():
    assert set(documented("Agent status")) == set(get_args(AgentStatus))
    kinds = source_strings("loop.py", r'AgentModelError\("([a-z_]+)"') | {"incomplete", "refusal"}
    assert 'return "incomplete"' in SOURCE["loop.py"] and 'return "refusal"' in SOURCE["loop.py"]
    assert set(documented("Agent error kinds")) == kinds


def test_review_codes():
    assert set(documented("Review status")) == set(get_args(ReviewStatus))
    assert set(documented("Review decisions")) == set(get_args(Resolution))


def test_run_codes_exist_in_the_code():
    assert set(documented("Run status")) == {"completed", "failed"}
    for status in ("completed", "failed"):
        assert f'"status": "{status}"' in SOURCE["investigate.py"] + SOURCE["agent_routes.py"]
    assert set(documented("Run error stages")) == {"extraction", "embedding", "coverage", "unexpected"}
    for stage in documented("Run error stages"):
        assert f'"{stage}"' in SOURCE["agent_routes.py"]


def test_tool_outcomes_are_the_errors_the_tools_and_the_loop_can_return():
    errors = set()
    for name in ("tools.py", "toolset.py", "loop.py"):
        errors |= source_strings(name, r'"error": "([a-z_]+)"')
    errors |= {f"document_{kind}" for kind in source_strings("documents.py", r'DocumentError\("([a-z_]+)"')}
    assert set(documented("Tool outcomes")) == errors | {"ok"}


def test_the_page_says_which_text_is_english_only():
    for field in ("findings", "summary", "rationale"):
        assert field in DOC.split("## Routes")[0]
