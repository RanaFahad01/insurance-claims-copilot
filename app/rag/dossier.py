"""Combine the rules decision and the coverage result into the final route, and package everything as a dossier.

The final route is decided here, in plain code. The best the system ever says is auto_approve_candidate, and only when
the rules, the coverage check and both gates all agree. Anything else goes to a person. It never says "approved"
or "denied": an excluded claim is flagged for a human, not rejected.
"""
from collections.abc import Mapping
from typing import Literal

from chromadb.api.models.Collection import Collection
from pydantic import BaseModel

from .. import extractor
from ..rules import triage
from ..schemas import ClaimExtraction, RouteDecision
from .coverage import CoverageModel, CoverageResult, assess_coverage
from .embeddings import Embedder
from .index import index_info
from .retriever import DISTANCE_CUTOFF, RetrievedClause, retrieve_context

FinalRoute = Literal["auto_approve_candidate", "escalate"]

# A gate always produces cannot_determine, so a gate reason replaces coverage_undetermined instead of repeating it.
GATE_REASONS = {
    "low_retrieval_confidence": "retrieval_low_confidence",
    "citation_check_failed": "citation_check_failed",
}


class Versions(BaseModel):
    """Everything needed to say which setup produced a dossier."""
    extraction_model: str
    extraction_prompt: str
    coverage_model: str
    coverage_prompt: str
    embedding_model: str
    policy_hash: str
    distance_cutoff: float


class Dossier(BaseModel):
    extraction: ClaimExtraction
    rules_decision: RouteDecision
    coverage: CoverageResult
    final_route: FinalRoute
    reasons: list[str]
    versions: Versions


def final_route(rules_decision: RouteDecision, coverage: CoverageResult) -> tuple[FinalRoute, list[str]]:
    """Escalate when any trigger fires. The Phase 2 reasons come first and are carried over unchanged."""
    reasons = list(rules_decision.reasons)

    if coverage.gate_triggered != "none":
        reasons.append(GATE_REASONS[coverage.gate_triggered])
    elif coverage.status == "excluded":
        reasons.append("coverage_possible_exclusion")
    elif coverage.status == "cannot_determine":
        reasons.append("coverage_undetermined")

    # A result that is both covered and gated should not exist, but the gate still wins if it ever does.
    escalate = rules_decision.route == "escalate" or coverage.status != "covered" or coverage.gate_triggered != "none"
    return ("escalate" if escalate else "auto_approve_candidate"), reasons


def collect_versions(
    coverage_model: CoverageModel,
    index_details: Mapping[str, str | int],
    cutoff: float = DISTANCE_CUTOFF,
) -> Versions:
    """The embedding model and policy hash come from the index, because that is what was actually searched."""
    return Versions(
        extraction_model=extractor.MODEL,
        extraction_prompt=extractor.PROMPT_VERSION,
        coverage_model=coverage_model.model,
        coverage_prompt=coverage_model.prompt_version,
        embedding_model=str(index_details["embedding_model"]),
        policy_hash=str(index_details["policy_hash"]),
        distance_cutoff=cutoff,
    )


def build_dossier(
    extraction: ClaimExtraction,
    rules_decision: RouteDecision,
    coverage: CoverageResult,
    versions: Versions,
) -> Dossier:
    route, reasons = final_route(rules_decision, coverage)
    return Dossier(
        extraction=extraction,
        rules_decision=rules_decision,
        coverage=coverage,
        final_route=route,
        reasons=reasons,
        versions=versions,
    )


def review_claim(
    extraction: ClaimExtraction,
    claim_text: str,
    collection: Collection,
    embedder: Embedder,
    coverage_model: CoverageModel,
    cutoff: float = DISTANCE_CUTOFF,
) -> Dossier:
    """Rules, retrieval, coverage and final routing for one claim. Retrieval searches with the extraction's summary;
    the coverage step reads the claim text as well."""
    rules_decision = triage(extraction)
    retrieved = retrieve_context(collection, embedder, extraction)
    coverage = assess_coverage(extraction, claim_text, retrieved, coverage_model, cutoff=cutoff)
    versions = collect_versions(coverage_model, index_info(collection), cutoff)
    return build_dossier(extraction, rules_decision, coverage, versions)


def _distance_note(clause: RetrievedClause) -> str:
    note = ", added because it is always checked" if clause.added_by_rule else ""
    return f"(distance {clause.distance:.2f}{note})"


def _clause_block(clause: RetrievedClause) -> str:
    return f"**[{clause.chunk_id}] {clause.title}** {_distance_note(clause)}\n{clause.text}"


def render_markdown(dossier: Dossier) -> str:
    """A short plain-text version of the dossier for a person to read."""
    e, c, v = dossier.extraction, dossier.coverage, dossier.versions
    amount = "not stated" if e.estimated_amount is None else f"{e.estimated_amount:g} {e.currency or ''}".strip()
    route = dossier.final_route.replace("_", " ")

    cited_ids = {clause.chunk_id for clause in c.cited_clauses}
    others = [clause for clause in c.retrieved if clause.chunk_id not in cited_ids]

    lines = [
        "# Claim review",
        "",
        f"**Recommendation: {route}** (rules severity score {dossier.rules_decision.severity_score})",
        f"Reasons: {', '.join(dossier.reasons) if dossier.reasons else 'none'}",
        "",
        "## Extracted facts",
        f"- Claimant: {e.claimant_name or 'not stated'}",
        f"- Policy number: {e.policy_number or 'not stated'}",
        f"- Incident date: {e.incident_date or 'not stated'}",
        f"- Incident location: {e.incident_location or 'not stated'}",
        f"- Damage type: {e.damage_type}",
        f"- Amount: {amount}",
        f"- Summary: {e.summary}",
        "",
        "## Coverage check",
        f"Status: **{c.status}** (gate triggered: {c.gate_triggered})",
        c.rationale,
        "",
        "### Cited clauses",
    ]
    lines += [_clause_block(clause) + "\n" for clause in c.cited_clauses] or ["None.\n"]
    if others:
        lines += ["### Other clauses the model saw"]
        lines += [f"- [{clause.chunk_id}] {clause.title} {_distance_note(clause)}" for clause in others]
        lines += [""]
    lines += [
        "## Versions",
        f"Extraction: {v.extraction_model}, {v.extraction_prompt}. Coverage: {v.coverage_model}, {v.coverage_prompt}.",
        f"Embeddings: {v.embedding_model}. Policy hash: {v.policy_hash[:12]}. Distance cutoff: {v.distance_cutoff}.",
    ]
    return "\n".join(lines) + "\n"
