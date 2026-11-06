"""A record of what the agent did, one entry per tool call.

For now the records are kept in memory. Writing them to a file with a run ID comes later, behind the same interface.
"""
from pydantic import BaseModel

CLIP_CHARS = 500


def clip(text: str, limit: int = CLIP_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class ToolCallRecord(BaseModel):
    step: int
    tool: str
    arguments: str                # exactly what the model sent, clipped
    outcome: str                  # "ok", or the error the model was shown
    result: str                   # the JSON the model was shown, clipped
    latency_ms: int
    side_effect: bool
    detail: str | None = None     # for us only, never shown to the model (for example the exception type)


class TurnRecord(BaseModel):
    """One call to the model."""
    step: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    tool_calls: int               # how many function calls the model asked for in this turn


class Trace:
    def __init__(self) -> None:
        self.step = 0             # the loop sets this before each model turn
        self.records: list[ToolCallRecord] = []
        self.turns: list[TurnRecord] = []

    def record(
        self,
        *,
        tool: str,
        arguments: str,
        outcome: str,
        result: str,
        latency_ms: int,
        side_effect: bool,
        detail: str | None = None,
    ) -> ToolCallRecord:
        entry = ToolCallRecord(
            step=self.step,
            tool=clip(tool, 80),
            arguments=clip(arguments),
            outcome=outcome,
            result=clip(result),
            latency_ms=latency_ms,
            side_effect=side_effect,
            detail=detail,
        )
        self.records.append(entry)
        return entry
