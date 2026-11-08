"""The whole flow for one claim: Phase 3's review, then the bounded agent step, then the deterministic final route.

    mandatory pipeline (code):  extraction (done by the caller) -> rules -> retrieval -> coverage
    bounded agent (only if documents are attached): compare_documents and the discretionary lookups
    finalize (code): the agent can only add caution
    escalation (code): a pending review record for every claim that goes to a person

The agent is never given the raw claim text, and its text never reaches the routing or the review record.
"""
import hashlib
import json
import time
from collections.abc import Callable
from types import SimpleNamespace

from pydantic import BaseModel

from ..rag.coverage import CoverageModel
from ..rag.dossier import Dossier, FinalRoute, review_claim
from ..rag.retriever import DISTANCE_CUTOFF
from ..schemas import ClaimExtraction
from .finalize import finalize
from .loop import DEFAULT_LIMITS, AgentModel, AgentResult, Limits, run_agent
from .policy_terms import PayoutResult
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_task
from .state import InvestigationState, Mismatch, RunContext
from .tools import Registry, execute_tool
from .toolset import ToolDeps, default_registry
from .trace import Trace, ToolCallRecord


def claim_id_for(claim_text: str) -> str:
    """The same claim text always gives the same ID, so submitting it twice does not create a second review."""
    return hashlib.sha256(claim_text.strip().encode("utf-8")).hexdigest()[:12]


class AgentVersions(BaseModel):
    agent_model: str
    agent_prompt: str
    invoice_model: str
    invoice_prompt: str


class Investigation(BaseModel):
    claim_id: str
    dossier: Dossier                          # the Phase 3 result, unchanged
    documents_attached: list[str]
    documents_compared: list[str]
    mismatches: list[Mismatch]
    payout: PayoutResult | None               # information for the reviewer, not used when routing
    findings: str | None                      # written by the AI agent; shown to the reviewer, never used when routing
    agent: AgentResult | None                 # None when no documents were attached, so the agent was not run
    tool_calls: list[ToolCallRecord]
    final_route: FinalRoute
    reasons: list[str]
    review_created: bool | None               # None when the claim was not escalated (or the record could not be written)
    review_error: str | None
    agent_versions: AgentVersions


def escalation_summary(extraction: ClaimExtraction, reasons: list[str]) -> str:
    """Built by code from the reasons and the extraction. Nothing the agent wrote goes into it."""
    return f"Sent to a person because: {', '.join(reasons)}. Claim: {extraction.summary}"


def investigate_claim(
    extraction: ClaimExtraction,
    claim_text: str,
    document_ids: list[str],
    deps: ToolDeps,
    coverage_model: CoverageModel,
    agent_model: AgentModel,
    registry: Registry | None = None,
    limits: Limits = DEFAULT_LIMITS,
    trace: Trace | None = None,
    clock: Callable[[], float] = time.monotonic,
    cutoff: float = DISTANCE_CUTOFF,
) -> Investigation:
    registry = registry if registry is not None else default_registry()
    trace = trace if trace is not None else Trace()

    dossier = review_claim(extraction, claim_text, deps.collection, deps.embedder, coverage_model, cutoff)
    state = InvestigationState(claim_id=claim_id_for(claim_text), document_ids=list(document_ids))
    ctx = RunContext(extraction=extraction, state=state, deps=deps)

    agent_result = None
    if state.documents_attached:
        task = build_task(extraction, dossier, state.document_ids)
        agent_result = run_agent(agent_model, SYSTEM_PROMPT, task, registry, ctx, trace, limits, clock)

    route, reasons = finalize(dossier.final_route, dossier.reasons, state)

    review_created, review_error = None, None
    if route == "escalate":
        trace.step = 0       # not a model turn: the surrounding code is making this call
        call = SimpleNamespace(
            name="escalate_to_human", call_id="envelope",
            arguments=json.dumps({"summary": escalation_summary(extraction, reasons), "reasons": reasons}),
        )
        outcome = execute_tool(call, registry, ctx, trace, from_model=False)
        if "created" in outcome:
            review_created = outcome["created"]
        else:
            review_error = outcome.get("error", "unknown")

    return Investigation(
        claim_id=state.claim_id,
        dossier=dossier,
        documents_attached=list(state.document_ids),
        documents_compared=list(state.compared_document_ids),
        mismatches=list(state.mismatches),
        payout=state.payout,
        findings=state.agent_text,
        agent=agent_result,
        tool_calls=list(trace.records),
        final_route=route,
        reasons=reasons,
        review_created=review_created,
        review_error=review_error,
        agent_versions=AgentVersions(
            agent_model=agent_model.model, agent_prompt=PROMPT_VERSION,
            invoice_model=deps.invoice_model.model, invoice_prompt=deps.invoice_model.prompt_version,
        ),
    )
