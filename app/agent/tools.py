"""Tools the agent can call, and the one safe way to run them.

The model only ever produces a name and a JSON string of arguments. Everything else happens here, in code: the arguments
are checked, the function runs, and whatever goes wrong comes back to the model as data. execute_tool never raises.

A tool that updates the state should do it as its last step, so a failure cannot leave the state half changed.
"""
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .state import RunContext
from .trace import Trace

MAX_ARGUMENTS_CHARS = 2000
MAX_RESULT_CHARS = 4000
MAX_ERROR_FIELDS = 5
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ToolArgs(BaseModel):
    """Base class for tool arguments. An argument the tool does not declare is an error, not ignored."""
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[ToolArgs]
    fn: Callable[[RunContext, Any], dict]
    side_effect: bool = False
    model_visible: bool = True        # False: only the surrounding code may run it


Registry = Mapping[str, Tool]


def build_registry(tools: Iterable[Tool]) -> dict[str, Tool]:
    registry: dict[str, Tool] = {}
    for tool in tools:
        if not NAME_PATTERN.fullmatch(tool.name):
            raise ValueError(f"invalid tool name: {tool.name!r}")
        if tool.name in registry:
            raise ValueError(f"duplicate tool name: {tool.name}")
        if not tool.description.strip():
            raise ValueError(f"tool {tool.name} needs a description")
        if tool.side_effect and tool.model_visible:
            raise ValueError(f"tool {tool.name} writes something, so the model may not call it")
        registry[tool.name] = tool
    return registry


def _close_objects(node: Any) -> None:
    """Strict mode wants every object closed and every property listed as required, at every level."""
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for value in node.values():
            _close_objects(value)
    elif isinstance(node, list):
        for value in node:
            _close_objects(value)


def to_function_schema(tool: Tool) -> dict:
    schema = tool.args_model.model_json_schema()
    _close_objects(schema)
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
        "strict": True,
    }


def model_schemas(registry: Registry) -> list[dict]:
    """The tool list sent to the model. Tools it may not call are not in it."""
    return [to_function_schema(t) for t in registry.values() if t.model_visible]


def _error_fields(error: ValidationError) -> list[dict]:
    """Which fields were wrong and why, without repeating what the model sent."""
    fields = []
    for item in error.errors()[:MAX_ERROR_FIELDS]:
        path = ".".join(str(p) for p in item["loc"]) or "(root)"
        fields.append({"field": path[:60], "problem": item["type"]})
    return fields


def _run(tool: Tool, raw_arguments: str, ctx: RunContext) -> tuple[dict, str | None, bool]:
    """Returns the result, a note for the trace that the model does not see, and whether the function was run."""
    if not isinstance(raw_arguments, str):
        return {"error": "invalid_arguments", "fields": [{"field": "(root)", "problem": "not_valid_json"}]}, None, False
    if len(raw_arguments) > MAX_ARGUMENTS_CHARS:
        return {"error": "arguments_too_long"}, None, False
    try:
        data = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        return {"error": "invalid_arguments", "fields": [{"field": "(root)", "problem": "not_valid_json"}]}, None, False
    try:
        args = tool.args_model.model_validate(data)
    except ValidationError as e:
        return {"error": "invalid_arguments", "fields": _error_fields(e)}, None, False

    try:
        result = tool.fn(ctx, args)
    except Exception as e:
        return {"error": "tool_failed"}, type(e).__name__, True
    if not isinstance(result, dict):
        return {"error": "tool_failed"}, f"returned {type(result).__name__}, not a dict", True
    try:
        text = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        return {"error": "tool_failed"}, f"result not JSON: {type(e).__name__}", True
    if len(text) > MAX_RESULT_CHARS:
        return {"error": "result_too_large"}, f"{len(text)} characters", True
    return result, None, True


def execute_tool(call: Any, registry: Registry, ctx: RunContext, trace: Trace, *, from_model: bool = True) -> dict:
    """Run one function call from the model. `call` needs .name and .arguments (the Responses API item has both)."""
    started = perf_counter()
    tool = registry.get(call.name)
    detail, ran = None, False
    if tool is None or (from_model and not tool.model_visible):
        # A hidden tool answers exactly like a missing one, and its function is never run.
        result = {"error": "unknown_tool"}
    else:
        result, detail, ran = _run(tool, call.arguments, ctx)

    trace.record(
        tool=str(call.name),
        arguments=str(call.arguments),
        outcome=str(result.get("error", "ok")),
        result=json.dumps(result, ensure_ascii=False),
        latency_ms=round((perf_counter() - started) * 1000),
        side_effect=bool(ran and tool.side_effect),   # whether a writing function actually ran
        detail=detail,
    )
    return result
