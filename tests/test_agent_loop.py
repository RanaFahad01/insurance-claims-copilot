import json

import pytest

from app.agent.loop import DEFAULT_LIMITS, AgentModelError, Limits, _signature, run_agent
from app.agent.state import InvestigationState, RunContext
from app.agent.tools import Tool, ToolArgs, build_registry
from app.agent.trace import Trace
from app.schemas import ClaimExtraction

from fakes import FakeAgentModel, agent_response, fn_call


class EchoArgs(ToolArgs):
    text: str


class NoArgs(ToolArgs):
    pass


class Counter:
    def __init__(self):
        self.runs = 0

    def echo(self, ctx, args):
        self.runs += 1
        return {"echo": args.text}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def counter():
    return Counter()


@pytest.fixture
def registry(counter):
    def escalate(ctx, args):
        counter.runs += 100
        return {"escalated": True}

    return build_registry([
        Tool("echo", "Repeat some text.", EchoArgs, counter.echo),
        Tool("escalate", "Writes a review record.", NoArgs, escalate, side_effect=True, model_visible=False),
    ])


def make_ctx(document_ids=()) -> RunContext:
    extraction = ClaimExtraction(
        incident_date=None, incident_location=None, claimant_name=None, policy_number=None, damage_type="water",
        estimated_amount=None, currency=None, language="en", summary="A pipe burst.",
    )
    return RunContext(extraction=extraction, state=InvestigationState(claim_id="c1", document_ids=list(document_ids)))


def go(model, registry, limits=DEFAULT_LIMITS, clock=None, ctx=None):
    ctx = ctx or make_ctx()
    trace = Trace()
    kwargs = {"clock": clock} if clock else {}
    result = run_agent(model, "system prompt", "the task", registry, ctx, trace, limits, **kwargs)
    return result, ctx, trace


def echo(text: str, call_id: str):
    return fn_call("echo", json.dumps({"text": text}), call_id)


def outputs_in(request) -> dict[str, dict]:
    return {i["call_id"]: json.loads(i["output"]) for i in request.input
            if isinstance(i, dict) and i.get("type") == "function_call_output"}


# --- the normal paths ------------------------------------------------------------------------------------------

def test_a_model_that_answers_at_once_finishes_in_one_step(registry):
    model = FakeAgentModel([agent_response(text="No documents to check.")])
    result, ctx, trace = go(model, registry)
    assert result.status == "finished" and result.error_kind is None
    assert result.text == "No documents to check."
    assert result.steps == 1 and result.tool_calls == 0
    assert ctx.state.agent_status == "finished" and ctx.state.steps == 1 and ctx.state.agent_text == "No documents to check."
    assert trace.records == []


def test_the_first_request_has_the_prompt_the_task_and_only_visible_tools(registry):
    model = FakeAgentModel([agent_response(text="done")])
    go(model, registry)
    request = model.requests[0]
    assert request.input == [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "the task"}]
    assert [t["name"] for t in request.tools] == ["echo"]


def test_a_tool_call_is_run_and_its_result_goes_back_with_the_same_call_id(registry, counter):
    model = FakeAgentModel([agent_response(echo("hi", "call_1")), agent_response(text="It said hi.")])
    result, ctx, trace = go(model, registry)
    assert result.status == "finished" and result.steps == 2 and result.tool_calls == 1
    assert counter.runs == 1
    assert outputs_in(model.requests[1]) == {"call_1": {"echo": "hi"}}
    assert [r.tool for r in trace.records] == ["echo"] and trace.records[0].step == 1
    assert ctx.state.tool_calls == 1


def test_parallel_calls_are_all_run_in_order_and_each_is_answered(registry, counter):
    model = FakeAgentModel([agent_response(echo("a", "call_a"), echo("b", "call_b")), agent_response(text="ok")])
    result, _, trace = go(model, registry)
    assert result.status == "finished" and result.tool_calls == 2
    assert outputs_in(model.requests[1]) == {"call_a": {"echo": "a"}, "call_b": {"echo": "b"}}
    assert [r.arguments for r in trace.records] == ['{"text": "a"}', '{"text": "b"}']


def test_more_calls_than_the_turn_limit_are_answered_but_not_run(registry, counter):
    calls = [echo(str(i), f"call_{i}") for i in range(6)]
    model = FakeAgentModel([agent_response(*calls), agent_response(text="ok")])
    result, _, trace = go(model, registry)
    assert counter.runs == DEFAULT_LIMITS.max_calls_per_turn == 4
    answers = outputs_in(model.requests[1])
    assert len(answers) == 6
    assert answers["call_4"] == answers["call_5"] == {"error": "skipped_limit_reached"}
    assert result.tool_calls == 4
    assert [r.outcome for r in trace.records[-2:]] == ["skipped_limit_reached"] * 2


def test_a_hidden_tool_named_by_the_model_is_not_run(registry, counter):
    model = FakeAgentModel([agent_response(fn_call("escalate", "{}", "call_1")), agent_response(text="ok")])
    result, _, _ = go(model, registry)
    assert counter.runs == 0
    assert outputs_in(model.requests[1]) == {"call_1": {"error": "unknown_tool"}}
    assert result.status == "finished"


def test_bad_arguments_come_back_as_data_and_the_run_continues(registry, counter):
    model = FakeAgentModel([agent_response(fn_call("echo", "not json", "call_1")), agent_response(text="ok")])
    result, _, _ = go(model, registry)
    assert outputs_in(model.requests[1])["call_1"]["error"] == "invalid_arguments"
    assert counter.runs == 0 and result.status == "finished"


def test_tokens_and_turns_are_counted(registry):
    model = FakeAgentModel([
        agent_response(echo("a", "call_1"), input_tokens=100, output_tokens=20),
        agent_response(text="ok", input_tokens=150, output_tokens=10),
    ])
    result, _, trace = go(model, registry)
    assert (result.input_tokens, result.output_tokens) == (250, 30)
    assert [(t.step, t.input_tokens, t.tool_calls) for t in result.turns] == [(1, 100, 1), (2, 150, 0)]
    assert trace.turns == result.turns


def test_long_final_text_is_clipped(registry):
    model = FakeAgentModel([agent_response(text="x" * 10_000)])
    result, ctx, _ = go(model, registry)
    assert len(result.text) < 4100 and result.text.endswith("...")
    assert ctx.state.agent_text == result.text


# --- limits ----------------------------------------------------------------------------------------------------

def test_a_model_that_never_stops_hits_the_step_limit(registry, counter):
    model = FakeAgentModel([agent_response(echo(str(i), f"call_{i}")) for i in range(10)])
    result, ctx, _ = go(model, registry, Limits(max_steps=3))
    assert result.status == "max_steps_exceeded" and result.steps == 3
    assert len(model.requests) == 3
    assert counter.runs == 3
    assert ctx.state.agent_status == "max_steps_exceeded"


def test_the_token_budget_stops_the_run_but_the_last_calls_still_ran(registry, counter):
    model = FakeAgentModel([agent_response(echo("a", "call_1"), input_tokens=120), agent_response(text="never reached")])
    result, ctx, _ = go(model, registry, Limits(max_total_tokens=100))
    assert result.status == "budget_exceeded" and result.steps == 1
    assert counter.runs == 1
    assert len(model.requests) == 1
    assert ctx.state.agent_status == "budget_exceeded"


def test_the_deadline_stops_the_run(registry):
    clock = FakeClock()
    model = FakeAgentModel(
        [agent_response(echo("a", "call_1")), agent_response(text="never reached")],
        on_create=lambda n: setattr(clock, "now", clock.now + 100),
    )
    result, _, _ = go(model, registry, Limits(deadline_seconds=60), clock=clock)
    assert result.status == "deadline_exceeded" and result.steps == 1
    assert result.elapsed_ms == 100_000


def test_each_request_gets_the_smaller_of_the_request_timeout_and_the_time_left(registry):
    clock = FakeClock()
    model = FakeAgentModel(
        [agent_response(echo("a", "call_1")), agent_response(text="ok")],
        on_create=lambda n: setattr(clock, "now", clock.now + 25),
    )
    go(model, registry, Limits(deadline_seconds=40, request_timeout_seconds=30), clock=clock)
    assert [r.timeout_s for r in model.requests] == [30, 15]


def test_repeating_an_identical_call_is_answered_without_running_it(registry, counter):
    model = FakeAgentModel([
        agent_response(echo("same", "call_1")),
        agent_response(echo("same", "call_2")),
        agent_response(text="ok"),
    ])
    result, _, trace = go(model, registry)
    assert counter.runs == 1
    assert outputs_in(model.requests[2])["call_2"]["error"] == "duplicate_call"
    assert result.status == "finished" and result.tool_calls == 1
    assert [r.outcome for r in trace.records] == ["ok", "duplicate_call"]


def test_a_model_that_keeps_repeating_a_call_is_stopped_as_looping(registry, counter):
    model = FakeAgentModel([agent_response(echo("same", f"call_{i}")) for i in range(6)])
    result, ctx, _ = go(model, registry)
    assert result.status == "looping"
    assert result.steps == 3                      # the call, then two repeats
    assert counter.runs == 1
    assert ctx.state.agent_status == "looping"


def test_the_same_arguments_in_a_different_order_count_as_a_repeat():
    a = fn_call("t", '{"a": 1, "b": 2}', "c1")
    b = fn_call("t", '{"b":2,"a":1}', "c2")
    assert _signature(a) == _signature(b)
    assert _signature(a) != _signature(fn_call("t", '{"a": 1, "b": 3}', "c3"))
    assert _signature(a) != _signature(fn_call("other", '{"a": 1, "b": 2}', "c4"))
    assert _signature(fn_call("t", "not json", "c5")) == "t:not json"


# --- problems with the model -----------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["rate_limited", "timeout", "connection", "api_error", "bad_request"])
def test_a_model_error_ends_the_run_with_an_error_status_and_does_not_raise(registry, kind):
    model = FakeAgentModel([AgentModelError(kind, "simulated")])
    result, ctx, _ = go(model, registry)
    assert result.status == "error" and result.error_kind == kind
    assert ctx.state.agent_status == "error"


def test_an_error_in_the_middle_of_a_run_keeps_what_the_tools_already_did(registry, counter):
    model = FakeAgentModel([agent_response(echo("a", "call_1")), AgentModelError("timeout", "simulated")])
    result, ctx, _ = go(model, registry)
    assert result.status == "error" and result.tool_calls == 1 and result.steps == 2
    assert ctx.state.tool_calls == 1


def test_a_cut_off_response_is_an_error(registry):
    model = FakeAgentModel([agent_response(text="partial", status="incomplete")])
    result, _, _ = go(model, registry)
    assert result.status == "error" and result.error_kind == "incomplete"
    assert result.text is None


def test_a_refusal_is_an_error(registry):
    model = FakeAgentModel([agent_response(refusal="I cannot help with that.")])
    result, _, _ = go(model, registry)
    assert result.status == "error" and result.error_kind == "refusal"


def test_a_bug_in_the_script_is_not_hidden_by_the_loop(registry):
    model = FakeAgentModel([])
    with pytest.raises(AssertionError):
        go(model, registry)


# --- the history always stays valid ----------------------------------------------------------------------------

def test_every_request_in_a_long_run_has_each_call_paired_with_one_output(registry):
    # The fake model raises the API's own 400 if any call in the history is missing its output, so reaching the
    # end of the run is the check. Mix normal, repeated, over-limit, unknown and malformed calls.
    turns = [
        agent_response(*[echo(str(i), f"a{i}") for i in range(6)]),
        agent_response(echo("0", "b0"), fn_call("nope", "{}", "b1"), fn_call("echo", "{", "b2")),
        agent_response(text="done"),
    ]
    result, _, _ = go(FakeAgentModel(turns), registry)
    assert result.status == "finished"
