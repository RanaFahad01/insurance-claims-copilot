"""The agent loop: ask the model, run the tool calls it asks for, repeat, and always stop.

The loop has no opinion about the claim. It enforces limits (steps, tokens, time, calls per turn, repeated calls) and
records what happened. Whatever way a run ends, the result and the investigation state say how, and the loop never
raises for a model or API problem. A bug in our own code is not caught here on purpose.
"""
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from .state import AgentStatus, RunContext
from .tools import Registry, execute_tool, model_schemas
from .trace import Trace, TurnRecord, clip

load_dotenv()

AGENT_MODEL = "gpt-6-luna"
MAX_TEXT_CHARS = 4000


@dataclass(frozen=True)
class Limits:
    max_steps: int = 8                    # model turns
    max_total_tokens: int = 30_000        # input plus output, summed over all turns
    deadline_seconds: float = 60.0        # for the whole run
    request_timeout_seconds: float = 30.0  # for one model call
    max_calls_per_turn: int = 4
    duplicate_limit: int = 2              # identical repeated calls tolerated before the run is stopped as looping


DEFAULT_LIMITS = Limits()


class AgentModelError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class AgentModel(Protocol):
    model: str

    def create(self, input: list, tools: list[dict], timeout_s: float) -> Any:
        ...


class OpenAIAgentModel:
    """One turn with the Responses API. Settings follow what was measured in the tool-call lab (see phase4 notes):
    temperature 0 only works with reasoning effort none, and we keep the history ourselves, so nothing is stored."""

    def __init__(self, model: str = AGENT_MODEL, client: OpenAI | None = None, parallel_tool_calls: bool = True):
        self.model = model
        self.parallel_tool_calls = parallel_tool_calls
        self._client = client if client is not None else OpenAI(timeout=30.0, max_retries=2)

    def create(self, input: list, tools: list[dict], timeout_s: float) -> Any:
        try:
            return self._client.responses.create(
                model=self.model,
                input=input,
                tools=tools,
                parallel_tool_calls=self.parallel_tool_calls,
                temperature=0,
                reasoning={"effort": "none"},
                store=False,
                timeout=timeout_s,
            )
        except RateLimitError as e:
            raise AgentModelError("rate_limited", "Rate limit hit after retries") from e
        except APITimeoutError as e:
            raise AgentModelError("timeout", "Request timed out after retries") from e
        except APIConnectionError as e:
            raise AgentModelError("connection", "Could not reach the API") from e
        except BadRequestError as e:
            # Usually our own mistake, such as a tool call without its output, so it gets its own kind.
            raise AgentModelError("bad_request", f"The API rejected the request: {e.message[:200]}") from e
        except APIStatusError as e:
            raise AgentModelError("api_error", f"API returned status {e.status_code}") from e


class AgentResult(BaseModel):
    status: AgentStatus
    error_kind: str | None
    text: str | None              # what the model wrote at the end. Kept for the dossier, never used for routing.
    steps: int
    tool_calls: int               # calls handed to a tool (including ones that came back as an error)
    input_tokens: int
    output_tokens: int
    elapsed_ms: int
    turns: list[TurnRecord]


def _signature(call: Any) -> str:
    """The same tool with the same arguments, whatever the key order or spacing."""
    try:
        arguments = json.dumps(json.loads(call.arguments), sort_keys=True)
    except (TypeError, ValueError):
        arguments = str(call.arguments)
    return f"{call.name}:{arguments}"


def _problem(response: Any) -> str | None:
    if response.status == "incomplete":
        return "incomplete"
    for item in response.output:
        if item.type == "message" and any(part.type == "refusal" for part in item.content):
            return "refusal"
    return None


def _output_item(call: Any, result: dict) -> dict:
    return {"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result, ensure_ascii=False)}


def _skipped(call: Any, trace: Trace, outcome: str, result: dict) -> dict:
    trace.record(
        tool=str(call.name), arguments=str(call.arguments), outcome=outcome,
        result=json.dumps(result, ensure_ascii=False), latency_ms=0, side_effect=False,
    )
    return result


def run_agent(
    model: AgentModel,
    system_prompt: str,
    task: str,
    registry: Registry,
    ctx: RunContext,
    trace: Trace,
    limits: Limits = DEFAULT_LIMITS,
    clock: Callable[[], float] = time.monotonic,
) -> AgentResult:
    started = clock()
    items: list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task}]
    schemas = model_schemas(registry)
    seen_calls: set[str] = set()
    duplicates = steps = tool_calls = input_tokens = output_tokens = 0
    status: AgentStatus = "max_steps_exceeded"
    error_kind: str | None = None
    text: str | None = None

    while True:
        # Limits are checked here, between turns. The tool calls of the last turn have already run.
        if steps >= limits.max_steps:
            status = "max_steps_exceeded"
            break
        if input_tokens + output_tokens >= limits.max_total_tokens:
            status = "budget_exceeded"
            break
        remaining = limits.deadline_seconds - (clock() - started)
        if remaining <= 0:
            status = "deadline_exceeded"
            break

        steps += 1
        trace.step = steps
        turn_started = clock()
        try:
            response = model.create(items, schemas, min(limits.request_timeout_seconds, remaining))
        except AgentModelError as e:
            status, error_kind = "error", e.kind
            break

        usage = response.usage
        turn_input = usage.input_tokens if usage else 0
        turn_output = usage.output_tokens if usage else 0
        input_tokens += turn_input
        output_tokens += turn_output
        calls = [i for i in response.output if i.type == "function_call"]
        trace.turns.append(TurnRecord(
            step=steps, input_tokens=turn_input, output_tokens=turn_output,
            latency_ms=round((clock() - turn_started) * 1000), tool_calls=len(calls),
        ))

        problem = _problem(response)
        if problem:
            status, error_kind = "error", problem
            break
        if not calls:
            status, text = "finished", response.output_text
            break

        items.extend(response.output)
        for index, call in enumerate(calls):
            signature = _signature(call)
            if index >= limits.max_calls_per_turn:
                result = _skipped(call, trace, "skipped_limit_reached", {"error": "skipped_limit_reached"})
            elif signature in seen_calls:
                duplicates += 1
                result = _skipped(call, trace, "duplicate_call", {
                    "error": "duplicate_call",
                    "hint": "You already made this exact call. Use the earlier result.",
                })
            else:
                seen_calls.add(signature)
                tool_calls += 1
                result = execute_tool(call, registry, ctx, trace)
            items.append(_output_item(call, result))

        if duplicates >= limits.duplicate_limit:
            status = "looping"
            break

    ctx.state.agent_status = status
    ctx.state.steps = steps
    ctx.state.tool_calls = tool_calls
    ctx.state.agent_text = clip(text, MAX_TEXT_CHARS) if text is not None else None
    return AgentResult(
        status=status,
        error_kind=error_kind,
        text=ctx.state.agent_text,
        steps=steps,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_ms=round((clock() - started) * 1000),
        turns=list(trace.turns),
    )
