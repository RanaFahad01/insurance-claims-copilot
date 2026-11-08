"""The final route after the agent step, decided in plain code.

It starts from the Phase 3 result (rules plus coverage, already combined by final_route) and can only add reasons.
The agent's text is never an input: the decision reads the typed investigation state, which only tool code changes.

    fast-track stays possible only if Phase 3 said so AND, when documents are attached, the agent finished, every
    document was compared successfully, and no mismatch was found.

Nothing here can turn an escalation into a fast-track, whatever the agent did or wrote.
"""
from ..rag.dossier import FinalRoute
from .state import InvestigationState


def finalize(base_route: FinalRoute, base_reasons: list[str], state: InvestigationState) -> tuple[FinalRoute, list[str]]:
    reasons = list(base_reasons)

    if state.documents_attached:
        if state.agent_status != "finished":
            reasons.append("agent_incomplete")
        if not state.required_checks_done:
            reasons.append("required_check_missing")
        if state.mismatches:
            reasons.append("document_mismatch")

    # Any reason at all means a person looks at it, so a fast-track never carries a reason.
    escalate = base_route == "escalate" or bool(reasons)
    return ("escalate" if escalate else "auto_approve_candidate"), reasons
