"""The same investigation flow as investigate.py, written as a LangGraph StateGraph.

    START -> extract -> review -> [documents attached?] -> investigate -> finalize -> [escalate?] -> escalate -> END
                                          \\-- no ---------------------------^                          (-> human_review -> END)

The nodes call the plain functions that investigate_claim calls, so the graph is orchestration and adds no logic.
The bounded agent loop is one node (run_agent, unchanged). This version is not served by the API: tests/test_agent_graph.py
shows that it gives the same answers as investigate_claim. What it adds is a checkpoint after every node and, with
human_review=True, a pause for a person's decision that can be resumed later.

State holds data only. The index, the models and the trace stay outside it, in the closures below, because the state is
what the checkpointer serializes. The checkpointer rebuilds only the types listed in CHECKPOINT_TYPES.
"""
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict, get_args

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..extractor import extract_claim
from ..rag.coverage import CoverageModel
from ..rag.dossier import Dossier, review_claim
from ..rag.retriever import DISTANCE_CUTOFF
from ..schemas import ClaimExtraction
from .finalize import finalize
from .investigate import (
    Investigation,
    assemble_investigation,
    escalate_step,
    escalation_summary,
    investigate_step,
    start_state,
)
from .loop import DEFAULT_LIMITS, AgentModel, AgentResult, Limits
from .reviews import Resolution
from .state import InvestigationState
from .tools import Registry
from .toolset import ToolDeps, default_registry
from .trace import Trace, new_run_id

# The only classes a checkpoint may rebuild (nested models inside them are rebuilt by pydantic).
CHECKPOINT_TYPES = [
    ("app.schemas", "ClaimExtraction"),
    ("app.rag.dossier", "Dossier"),
    ("app.agent.state", "InvestigationState"),
    ("app.agent.loop", "AgentResult"),
]


def make_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))


class FlowState(TypedDict, total=False):
    claim_text: str
    document_ids: list[str]
    run_id: str | None
    extraction: ClaimExtraction
    dossier: Dossier
    investigation: InvestigationState
    agent_result: AgentResult | None
    route: str
    reasons: list[str]
    review_created: bool | None
    review_error: str | None
    human_decision: dict | None


@dataclass
class FlowResources:
    """What the nodes use. Not part of the state."""
    deps: ToolDeps
    coverage_model: CoverageModel
    agent_model: AgentModel
    extract: Callable[[str], ClaimExtraction] = extract_claim
    registry: Registry | None = None
    limits: Limits = DEFAULT_LIMITS
    trace: Trace = field(default_factory=Trace)
    clock: Callable[[], float] = time.monotonic
    cutoff: float = DISTANCE_CUTOFF


def build_graph(res: FlowResources, checkpointer=None, human_review: bool = False):
    registry = res.registry if res.registry is not None else default_registry()
    deps = res.deps

    def extract_node(state: FlowState) -> dict:
        return {"extraction": res.extract(state["claim_text"])}

    def review_node(state: FlowState) -> dict:
        dossier = review_claim(state["extraction"], state["claim_text"], deps.collection, deps.embedder, res.coverage_model, res.cutoff)
        return {"dossier": dossier, "investigation": start_state(state["claim_text"], state["document_ids"])}

    def investigate_node(state: FlowState) -> dict:
        investigation = state["investigation"]
        result = investigate_step(state["extraction"], state["dossier"], investigation, deps, res.agent_model, registry,
                                  res.limits, res.trace, res.clock, state.get("run_id"))
        return {"investigation": investigation, "agent_result": result}

    def finalize_node(state: FlowState) -> dict:
        route, reasons = finalize(state["dossier"].final_route, state["dossier"].reasons, state["investigation"])
        return {"route": route, "reasons": reasons}

    def escalate_node(state: FlowState) -> dict:
        created, error = escalate_step(state["extraction"], state["route"], state["reasons"], state["investigation"], deps,
                                       registry, res.trace, state.get("run_id"))
        return {"review_created": created, "review_error": error}

    def human_review_node(state: FlowState) -> dict:
        # When this node is resumed it starts again from the top, so everything before interrupt() must be repeatable.
        # It is: it only builds a payload. The review record was written by escalate_node, which is idempotent.
        investigation = state["investigation"]
        decision = interrupt({
            "claim_id": investigation.claim_id, "run_id": state.get("run_id"), "reasons": state["reasons"],
            "summary": escalation_summary(state["extraction"], state["reasons"]),
        })
        if not valid_decision(decision):
            # Do not raise: LangGraph keeps the resume value of a failed node and replays it, so an exception here would leave
            # the run stuck on the bad value. resume_graph checks first, so this is only reached by a direct graph.invoke.
            # The review stays pending and can still be resolved through the review queue.
            return {"human_decision": {"resolution": None, "recorded": False, "review_found": None, "error": "invalid_decision"}}
        review, recorded = deps.reviews.resolve(investigation.claim_id, decision["resolution"], decision.get("note"))
        return {"human_decision": {"resolution": decision["resolution"], "recorded": recorded, "review_found": review is not None}}

    def after_review(state: FlowState) -> str:
        return "investigate" if state["investigation"].documents_attached else "finalize"

    def after_finalize(state: FlowState) -> str:
        return "escalate" if state["route"] == "escalate" else END

    graph = StateGraph(FlowState)
    graph.add_node("extract", extract_node)
    graph.add_node("review", review_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("escalate", escalate_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "review")
    graph.add_conditional_edges("review", after_review, {"investigate": "investigate", "finalize": "finalize"})
    graph.add_edge("investigate", "finalize")
    graph.add_conditional_edges("finalize", after_finalize, {"escalate": "escalate", END: END})
    if human_review:
        graph.add_node("human_review", human_review_node)
        graph.add_edge("escalate", "human_review")
        graph.add_edge("human_review", END)
    else:
        graph.add_edge("escalate", END)
    return graph.compile(checkpointer=checkpointer)


def _config(run_id: str) -> dict:
    # One thread per run, not per claim: a second run on the same claim text must not start from the first run's final state.
    return {"configurable": {"thread_id": run_id}}


def run_graph(graph, claim_text: str, document_ids: list[str], run_id: str) -> dict:
    """Runs to the end, or to the human-review pause (then the result has an "__interrupt__" entry)."""
    return graph.invoke({"claim_text": claim_text, "document_ids": list(document_ids), "run_id": run_id}, _config(run_id))


def valid_decision(decision: object) -> bool:
    return isinstance(decision, dict) and decision.get("resolution") in get_args(Resolution) and isinstance(
        decision.get("note"), (str, type(None)))


def resume_graph(graph, run_id: str, resolution: str, note: str | None = None) -> dict:
    """Continues a paused run with a person's decision. A decision that is not valid raises ValueError before the graph is
    touched, so the run stays paused and can be resumed again with a correct one."""
    if not valid_decision({"resolution": resolution, "note": note}):
        raise ValueError("a decision needs a resolution of: " + ", ".join(get_args(Resolution)))
    return graph.invoke(Command(resume={"resolution": resolution, "note": note}), _config(run_id))


def final_investigation(values: dict[str, Any], res: FlowResources) -> Investigation:
    """The same Investigation that investigate_claim returns, built from the graph's final state."""
    return assemble_investigation(
        values["dossier"], values["investigation"], values.get("agent_result"), res.trace, values["route"], values["reasons"],
        values.get("review_created"), values.get("review_error"), res.deps, res.agent_model, values.get("run_id"),
    )


def investigate_with_graph(claim_text: str, document_ids: list[str], res: FlowResources, run_id: str | None = None,
                           checkpointer=None) -> Investigation:
    """The graph counterpart of investigate_claim (the extraction happens inside the graph)."""
    graph = build_graph(res, checkpointer=checkpointer if checkpointer is not None else make_checkpointer())
    values = graph.invoke({"claim_text": claim_text, "document_ids": list(document_ids), "run_id": run_id},
                          _config(run_id or new_run_id()))
    return final_investigation(values, res)
