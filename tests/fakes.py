"""Test doubles shared by the RAG tests. They need no network and no API key."""
import hashlib
import math
import re
from types import SimpleNamespace

from app.agent.loop import AgentModelError


class FakeEmbedder:
    """A deterministic bag-of-words embedder.

    Texts that share words get similar vectors, so retrieval tests behave sensibly,
    and the same text always gives the same vector. Python's own hash() changes between
    runs, so a stable md5 is used to pick each word's slot.
    """

    def __init__(self, model: str = "fake-embed-1", dims: int = 64):
        self.model = model
        self.dims = dims
        self.calls = 0
        self.texts_embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts_embedded += len(texts)
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in re.findall(r"\w+", text.lower()):
            slot = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dims
            vector[slot] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


# --- the agent -------------------------------------------------------------------------------------------------


def fn_call(name: str, arguments: str, call_id: str):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def agent_response(*calls, text: str = "", input_tokens: int = 10, output_tokens: int = 5, status: str = "completed",
                   refusal: str | None = None):
    """What the Responses API returns for one turn, with only the parts the loop reads."""
    output = list(calls)
    if text:
        output.append(SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)]))
    if refusal:
        output.append(SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal=refusal)]))
    return SimpleNamespace(
        output=output, output_text=text, status=status,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens), incomplete_details=None,
    )


def _kind_and_id(item):
    if isinstance(item, dict):
        return item.get("type"), item.get("call_id")
    return getattr(item, "type", None), getattr(item, "call_id", None)


def check_call_pairing(items: list) -> None:
    """The API answers 400 when a function_call has no output, or an output has no call. So does this."""
    calls = [cid for kind, cid in map(_kind_and_id, items) if kind == "function_call"]
    outputs = [cid for kind, cid in map(_kind_and_id, items) if kind == "function_call_output"]
    for cid in calls:
        if outputs.count(cid) != 1:
            raise AgentModelError("bad_request", f"No tool output found for function call {cid}.")
    for cid in outputs:
        if cid not in calls:
            raise AgentModelError("bad_request", f"No tool call found for function call output with call_id {cid}.")


class FakeAgentModel:
    """Plays back scripted turns. An exception in the script is raised instead of returned.
    on_create(request_number) runs before each answer, for example to move a fake clock."""

    def __init__(self, script, on_create=None):
        self.model = "fake-agent"
        self.script = list(script)
        self.on_create = on_create
        self.requests: list[SimpleNamespace] = []

    def create(self, input, tools, timeout_s):
        self.requests.append(SimpleNamespace(input=list(input), tools=tools, timeout_s=timeout_s))
        check_call_pairing(input)
        if self.on_create:
            self.on_create(len(self.requests))
        if not self.script:
            raise AssertionError("the scripted model ran out of turns")
        turn = self.script.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


class FakeInvoiceModel:
    """Returns one fixed reading (or raises one fixed error) and remembers which texts it was given."""

    def __init__(self, result):
        self.model = "fake-invoice"
        self.prompt_version = "fake"
        self.result = result
        self.texts: list[str] = []

    def extract(self, document_text):
        self.texts.append(document_text)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result
