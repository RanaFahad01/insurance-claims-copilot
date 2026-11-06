"""One check of the agent loop against the real OpenAI API, with one tiny tool.

Skipped by default. Run it with: pytest -m live tests/test_agent_live.py
It needs OPENAI_API_KEY and costs a fraction of a cent.
"""
import pytest

from app.agent.loop import OpenAIAgentModel, run_agent
from app.agent.state import InvestigationState, RunContext
from app.agent.tools import Tool, ToolArgs, build_registry
from app.agent.trace import Trace
from app.schemas import ClaimExtraction


class GetClauseArgs(ToolArgs):
    clause_id: str


def get_clause(ctx, args):
    if args.clause_id != "en-C2.1":
        return {"error": "clause_not_found"}
    ctx.state.note_clause(args.clause_id)
    return {"clause_id": "en-C2.1", "title": "Water damage: covered", "text": "We cover sudden escape of water from fixed pipes."}


@pytest.mark.live
def test_the_real_model_calls_the_tool_and_the_loop_finishes():
    extraction = ClaimExtraction(
        incident_date=None, incident_location=None, claimant_name=None, policy_number=None, damage_type="water",
        estimated_amount=None, currency=None, language="en", summary="A pipe burst.",
    )
    ctx = RunContext(extraction=extraction, state=InvestigationState(claim_id="live-1"))
    trace = Trace()
    registry = build_registry([Tool("get_clause", "Fetch one policy clause by its exact ID, for example 'en-C2.1'.", GetClauseArgs, get_clause)])

    result = run_agent(
        OpenAIAgentModel(), "You answer questions about a policy using the tools provided.",
        "What does clause en-C2.1 say? Use the tool, then answer in one sentence.", registry, ctx, trace,
    )

    assert result.status == "finished", result
    assert result.tool_calls == 1 and trace.records[0].tool == "get_clause" and trace.records[0].outcome == "ok"
    assert ctx.state.clause_ids_seen == ["en-C2.1"]
    assert result.text and result.input_tokens > 0
