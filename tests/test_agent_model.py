from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, RateLimitError

from app.agent.loop import AGENT_MODEL, AgentModelError, OpenAIAgentModel

REQUEST = httpx2.Request("POST", "http://example.invalid/v1/responses")


class FakeResponsesAPI:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else SimpleNamespace(output=[])
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def make_model(**kwargs):
    api = FakeResponsesAPI(**{k: v for k, v in kwargs.items() if k in ("response", "error")})
    options = {k: v for k, v in kwargs.items() if k not in ("response", "error")}
    return OpenAIAgentModel(client=SimpleNamespace(responses=api), **options), api


def test_the_request_uses_the_settings_measured_in_the_tool_call_lab():
    response = SimpleNamespace(output=["an item"])
    model, api = make_model(response=response)
    tools = [{"type": "function", "name": "t"}]
    items = [{"role": "user", "content": "hi"}]

    assert model.create(items, tools, 12.5) is response
    assert api.kwargs == {
        "model": AGENT_MODEL,
        "input": items,
        "tools": tools,
        "parallel_tool_calls": True,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "store": False,
        "timeout": 12.5,
    }


def test_parallel_calls_can_be_switched_off():
    model, api = make_model(parallel_tool_calls=False)
    model.create([], [], 5)
    assert api.kwargs["parallel_tool_calls"] is False


@pytest.mark.parametrize(
    "error, kind",
    [
        (RateLimitError("simulated", response=httpx2.Response(429, request=REQUEST), body=None), "rate_limited"),
        (APITimeoutError(request=REQUEST), "timeout"),
        (APIConnectionError(request=REQUEST), "connection"),
        (BadRequestError("No tool output found for function call call_1.", response=httpx2.Response(400, request=REQUEST), body=None), "bad_request"),
        (APIStatusError("simulated", response=httpx2.Response(401, request=REQUEST), body=None), "api_error"),
    ],
)
def test_api_errors_are_mapped_to_agent_model_errors(error, kind):
    model, _ = make_model(error=error)
    with pytest.raises(AgentModelError) as raised:
        model.create([], [], 5)
    assert raised.value.kind == kind


def test_a_bad_request_says_what_the_api_complained_about():
    error = BadRequestError("No tool output found for function call call_1.", response=httpx2.Response(400, request=REQUEST), body=None)
    model, _ = make_model(error=error)
    with pytest.raises(AgentModelError, match="No tool output found"):
        model.create([], [], 5)
