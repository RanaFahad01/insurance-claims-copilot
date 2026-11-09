"""The Phase 4 endpoints: investigate a claim with its documents, read a run's trace, and work the review queue.

The Phase 1-3 endpoints stay in main.py. Everything here is built from injected dependencies (see dependencies.py), so
the tests run the whole flow with fakes. A person resolving a review only records a decision: it never changes the
route, the reasons or the trace of the run that escalated the claim.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, StringConstraints

from .agent.investigate import Investigation, claim_id_for, investigate_claim, run_summary
from .agent.loop import AgentModel
from .agent.reviews import Resolution, Review, ReviewStatus, ReviewStore
from .agent.toolset import ToolDeps
from .agent.trace import Trace, TraceWriter, new_run_id
from .dependencies import (
    get_agent_model,
    get_coverage_model,
    get_review_store,
    get_tool_deps,
    get_trace_writer,
    http_error,
)
from .extractor import ExtractionError, extract_claim
from .rag.coverage import CoverageError, CoverageModel
from .rag.embeddings import EmbeddingError

router = APIRouter()

# One line per investigation: IDs, route, counts. Never the claim text and never anything a document said.
log = logging.getLogger("claims.investigate")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

DocumentId = Annotated[str, StringConstraints(pattern=r"^doc_\d{2,4}$")]


class InvestigateIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)
    document_ids: list[DocumentId] = Field(default_factory=list, max_length=5)


class ResolveIn(BaseModel):
    resolution: Resolution
    note: str | None = Field(default=None, max_length=500)


def _save_trace(writer: TraceWriter, run_id: str, trace: Trace, summary: dict) -> None:
    """Watching a run must never break it: if the file cannot be written the answer is still returned, and the problem is logged."""
    try:
        writer.write(run_id, trace, summary)
    except OSError as e:
        log.error("run_id=%s trace_not_written=%s", run_id, type(e).__name__)


def _failure_summary(claim_text: str, started_at: str, started: float, stage: str, kind: str) -> dict:
    return {"status": "failed", "claim_id": claim_id_for(claim_text), "started_at": started_at,
            "elapsed_ms": round((time.monotonic() - started) * 1000), "error_stage": stage, "error_kind": kind}


@router.post("/claims/investigate", response_model=Investigation, response_model_exclude={"tool_calls"})
def investigate(
    req: InvestigateIn,
    response: Response,
    deps: ToolDeps = Depends(get_tool_deps),   # first: a missing index fails before any call to a language model
    coverage_model: CoverageModel = Depends(get_coverage_model),
    agent_model: AgentModel = Depends(get_agent_model),
    writer: TraceWriter = Depends(get_trace_writer),
):
    unknown = [d for d in dict.fromkeys(req.document_ids) if not deps.documents.exists(d)]
    if unknown:   # a caller's mistake: no point spending model calls on it
        raise HTTPException(status_code=422, detail=f"unknown_document: {', '.join(unknown)}")

    run_id, started_at, started, trace = new_run_id(), datetime.now(timezone.utc).isoformat(timespec="seconds"), time.monotonic(), Trace()
    response.headers["X-Run-ID"] = run_id
    try:
        extraction = extract_claim(req.text)
        result = investigate_claim(extraction, req.text, req.document_ids, deps, coverage_model, agent_model,
                                   trace=trace, run_id=run_id)
    except (ExtractionError, EmbeddingError, CoverageError) as e:
        stage = {ExtractionError: "extraction", EmbeddingError: "embedding", CoverageError: "coverage"}[type(e)]
        _save_trace(writer, run_id, trace, _failure_summary(req.text, started_at, started, stage, e.kind))
        error = http_error(e.kind, e)
        error.headers = {"X-Run-ID": run_id}
        raise error from e
    except Exception as e:
        _save_trace(writer, run_id, trace, _failure_summary(req.text, started_at, started, "unexpected", type(e).__name__))
        raise

    elapsed_ms = round((time.monotonic() - started) * 1000)
    summary = run_summary(result, started_at, elapsed_ms)
    _save_trace(writer, run_id, trace, summary)
    log.info(
        "run_id=%s claim_id=%s route=%s agent=%s steps=%s tool_calls=%s tokens=%s elapsed_ms=%s reasons=%s",
        run_id, result.claim_id, result.final_route, summary["agent_status"], summary["steps"], summary["tool_calls"],
        summary["input_tokens"] + summary["output_tokens"], elapsed_ms, ",".join(result.reasons) or "none",
    )
    return result


@router.get("/runs/{run_id}")
def get_run(run_id: str, writer: TraceWriter = Depends(get_trace_writer)):
    """The trace of one run. An ID that is not a run ID gets the same answer as one that does not exist."""
    try:
        lines = writer.read(run_id)
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail="trace_unreadable") from e
    if lines is None:
        raise HTTPException(status_code=404, detail="No run with that ID.")
    return {
        "run_id": run_id,
        "run": next((line for line in reversed(lines) if line.get("type") == "run"), None),
        "events": [line for line in lines if line.get("type") != "run"],
    }


@router.get("/reviews", response_model=list[Review])
def list_reviews(status: ReviewStatus | None = None, store: ReviewStore = Depends(get_review_store)):
    return store.list(status)


@router.post("/reviews/{claim_id}/resolve", response_model=Review)
def resolve_review(claim_id: str, req: ResolveIn, store: ReviewStore = Depends(get_review_store)):
    review, resolved_now = store.resolve(claim_id, req.resolution, req.note)
    if review is None:
        raise HTTPException(status_code=404, detail="No review with that ID.")
    if not resolved_now:
        raise HTTPException(status_code=409, detail={"error": "already_resolved", "review": review.model_dump()})
    return review
