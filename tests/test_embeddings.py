from types import SimpleNamespace

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.rag.embeddings import BATCH_SIZE, EMBEDDING_MODEL, EmbeddingError, OpenAIEmbedder

REQUEST = httpx2.Request("POST", "http://example.invalid/v1/embeddings")


def status_error(cls, status):
    return cls("simulated", response=httpx2.Response(status, request=REQUEST), body=None)


class FakeEmbeddingsAPI:
    """Mimics client.embeddings.create. A text 't7' becomes the vector [7.0]."""

    def __init__(self, error=None, reverse=False, drop_one=False):
        self.calls: list[list[str]] = []
        self.error = error
        self.reverse = reverse
        self.drop_one = drop_one

    def create(self, model, input):
        self.calls.append(list(input))
        if self.error is not None:
            raise self.error
        data = [SimpleNamespace(index=i, embedding=[float(text[1:])]) for i, text in enumerate(input)]
        if self.drop_one:
            data = data[:-1]
        if self.reverse:
            data = list(reversed(data))
        return SimpleNamespace(data=data)


def make_embedder(**kwargs):
    api = FakeEmbeddingsAPI(**kwargs)
    return OpenAIEmbedder(client=SimpleNamespace(embeddings=api)), api


def test_default_model_is_the_configured_constant():
    embedder, _ = make_embedder()
    assert embedder.model == EMBEDDING_MODEL


def test_vectors_come_back_in_input_order():
    embedder, _ = make_embedder()
    assert embedder.embed(["t3", "t1", "t2"]) == [[3.0], [1.0], [2.0]]


def test_order_is_restored_when_the_api_returns_items_out_of_order():
    embedder, _ = make_embedder(reverse=True)
    assert embedder.embed(["t0", "t1", "t2"]) == [[0.0], [1.0], [2.0]]


def test_large_inputs_are_split_into_batches_and_order_is_kept():
    embedder, api = make_embedder()
    texts = [f"t{i}" for i in range(BATCH_SIZE * 2 + 50)]
    vectors = embedder.embed(texts)
    assert [len(call) for call in api.calls] == [BATCH_SIZE, BATCH_SIZE, 50]
    assert vectors == [[float(i)] for i in range(len(texts))]


def test_empty_input_makes_no_api_call():
    embedder, api = make_embedder()
    assert embedder.embed([]) == []
    assert api.calls == []


@pytest.mark.parametrize(
    "error, kind",
    [
        (status_error(RateLimitError, 429), "rate_limited"),
        (APITimeoutError(request=REQUEST), "timeout"),
        (APIConnectionError(request=REQUEST), "connection"),
        (status_error(APIStatusError, 401), "api_error"),
    ],
)
def test_api_errors_are_mapped_to_embedding_errors(error, kind):
    embedder, _ = make_embedder(error=error)
    with pytest.raises(EmbeddingError) as raised:
        embedder.embed(["t1"])
    assert raised.value.kind == kind


def test_wrong_number_of_vectors_is_an_error():
    embedder, _ = make_embedder(drop_one=True)
    with pytest.raises(EmbeddingError) as raised:
        embedder.embed(["t1", "t2"])
    assert raised.value.kind == "bad_response"


@pytest.mark.live
def test_real_api_returns_unit_length_vectors():
    vectors = OpenAIEmbedder().embed(["Water damage from a burst pipe is covered."])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1536
    assert abs(sum(x * x for x in vectors[0]) ** 0.5 - 1.0) < 0.01
