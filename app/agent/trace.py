"""A record of what the agent did, one entry per tool call, kept in memory and written to one JSONL file per run.

A file holds three kinds of line: "turn" (one call to the model: tokens, latency), "tool_call" (arguments and result,
already clipped) and one final "run" line with the totals. The claim text, the full document text and what the agent
wrote are never written. A clipped tool result can still hold short excerpts: the vendor, the document's remarks, or a
value that disagreed with the claim (for example a name). Treat a trace as sensitive. The file is written once, when the run ends, to a temporary name and then renamed, so a reader never sees half a file.
"""
import json
import os
import re
import uuid
from pathlib import Path

from pydantic import BaseModel

CLIP_CHARS = 500
DEFAULT_TRACE_DIR = Path(__file__).resolve().parents[2] / "runs" / "traces"
RUN_ID_PATTERN = re.compile(r"[0-9a-f]{16}")


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def valid_run_id(value: object) -> bool:
    return isinstance(value, str) and RUN_ID_PATTERN.fullmatch(value) is not None


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


def trace_lines(run_id: str, trace: Trace, summary: dict) -> list[dict]:
    """The lines of a run's file, in the order things happened: each model turn followed by the tool calls it caused.
    Calls that belong to no turn (the code's own escalation, step 0) come after the last turn."""
    lines: list[dict] = []
    placed: set[int] = set()
    for turn in trace.turns:
        lines.append({"type": "turn", **turn.model_dump()})
        for i, record in enumerate(trace.records):
            if record.step == turn.step and i not in placed:
                placed.add(i)
                lines.append({"type": "tool_call", **record.model_dump()})
    for i, record in enumerate(trace.records):
        if i not in placed:
            lines.append({"type": "tool_call", **record.model_dump()})
    lines.append({"type": "run", **summary})
    return [{"run_id": run_id, "seq": n, **line} for n, line in enumerate(lines)]


class TraceWriter:
    """One JSONL file per run in one folder. The file name is built only from a run ID that passed valid_run_id."""

    def __init__(self, directory: Path = DEFAULT_TRACE_DIR):
        self.directory = Path(directory)

    def path_for(self, run_id: str) -> Path:
        if not valid_run_id(run_id):
            raise ValueError("invalid run id")
        return self.directory / f"{run_id}.jsonl"

    def write(self, run_id: str, trace: Trace, summary: dict) -> Path:
        path = self.path_for(run_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in trace_lines(run_id, trace, summary))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
        return path

    def read(self, run_id: str) -> list[dict] | None:
        """The lines of a run, or None if the ID is not a valid run ID or there is no such run."""
        if not valid_run_id(run_id):
            return None
        path = self.path_for(run_id)
        if not path.is_file():
            return None
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
