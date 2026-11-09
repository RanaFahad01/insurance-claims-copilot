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
from .reviews import MAX_SUMMARY_CHARS
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
    run_id: str | None = None                 # set by the API; the trace of this run is under this ID


def run_summary(inv: Investigation, started_at: str, elapsed_ms: int) -> dict:
    """The last line of a run's trace. Totals and decisions only: no claim text, no document text, no agent text."""
    agent = inv.agent
    return {
        "status": "completed", "claim_id": inv.claim_id, "started_at": started_at, "elapsed_ms": elapsed_ms,
        "agent_status": agent.status if agent else "not_run",
        "agent_error_kind": agent.error_kind if agent else None,
        "steps": agent.steps if agent else 0,
        "tool_calls": agent.tool_calls if agent else 0,
        "input_tokens": agent.input_tokens if agent else 0,
        "output_tokens": agent.output_tokens if agent else 0,
        "agent_elapsed_ms": agent.elapsed_ms if agent else 0,
        "final_route": inv.final_route, "reasons": inv.reasons, "phase3_route": inv.dossier.final_route,
        "documents": inv.documents_attached, "documents_compared": inv.documents_compared,
        "mismatches": [{"document_id": m.document_id, "field": m.field, "severity": m.severity} for m in inv.mismatches],
        "review_created": inv.review_created, "review_error": inv.review_error,
        "versions": inv.agent_versions.model_dump(),
    }


def escalation_summary(extraction: ClaimExtraction, reasons: list[str]) -> str:
    """Built by code from the reasons and the extraction. Nothing the agent wrote goes into it.
    Always within the limit the escalation tool accepts: the extraction's summary has no length limit of its own, and a claim
    must never fail to reach the review queue because a sentence was long."""
    text = f"Sent to a person because: {', '.join(reasons)}. Claim: {extraction.summary}"
    return text if len(text) <= MAX_SUMMARY_CHARS else text[:MAX_SUMMARY_CHARS - 3] + "..."


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
    run_id: str | None = None,
) -> Investigation:
    registry = registry if registry is not None else default_registry()
    trace = trace if trace is not None else Trace()

    dossier = review_claim(extraction, claim_text, deps.collection, deps.embedder, coverage_model, cutoff)
    state = InvestigationState(claim_id=claim_id_for(claim_text), document_ids=list(document_ids))
    ctx = RunContext(extraction=extraction, state=state, deps=deps, run_id=run_id)

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
        run_id=run_id,
    )
