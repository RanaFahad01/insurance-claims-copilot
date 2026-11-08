"""What the investigator agent is told.

agent-v1 has no defence against text that talks to the model, on purpose: the documents and the coverage rationale are
untrusted input, and the injection hardening comes later so the before-and-after stays clean. What keeps the system
safe is not this prompt but the code around it (finalize.py): the agent can only add caution.
"""
from ..rag.dossier import Dossier
from ..schemas import ClaimExtraction

PROMPT_VERSION = "agent-v1"

SYSTEM_PROMPT = """You are an investigator assisting an insurance claims adjuster.
You receive the extracted facts of a claim, the result of its coverage check, and the IDs of attached supporting documents.
Use the tools to check the attached documents against the claim. Check every document with compare_documents.
If the documents agree with the claim, use calculate_payout to state the expected payment. If a document or the claim raises a question about the policy (for example a document dated long after the incident, or an amount near a limit), look up the relevant clause or terms.
Report factual findings only. You cannot approve or deny a claim, and you cannot change its routing. When you are done, reply with a short list of findings.
"""


def _or_not_stated(value: object) -> str:
    return "not stated" if value is None else str(value)


def build_task(extraction: ClaimExtraction, dossier: Dossier, document_ids: list[str]) -> str:
    """The task message. It is built from the extraction and the earlier results, not from the raw claim text."""
    amount = "not stated" if extraction.estimated_amount is None else f"{extraction.estimated_amount:g} {extraction.currency or ''}".strip()
    coverage = dossier.coverage
    cited = ", ".join(c.chunk_id for c in coverage.cited_clauses) or "none"
    reasons = ", ".join(dossier.rules_decision.reasons) or "none"
    return "\n".join([
        "Claim facts:",
        f"- Claimant: {_or_not_stated(extraction.claimant_name)}",
        f"- Policy number: {_or_not_stated(extraction.policy_number)}",
        f"- Incident date: {_or_not_stated(extraction.incident_date)}",
        f"- Incident location: {_or_not_stated(extraction.incident_location)}",
        f"- Damage type: {extraction.damage_type}",
        f"- Amount: {amount}",
        f"- Summary: {extraction.summary}",
        "",
        "Coverage check (made earlier):",
        f"- Status: {coverage.status}",
        f"- Rationale: {coverage.rationale}",
        f"- Clauses cited: {cited}",
        f"- Rules route: {dossier.rules_decision.route} (reasons: {reasons})",
        "",
        f"Attached documents: {', '.join(document_ids)}",
    ])
