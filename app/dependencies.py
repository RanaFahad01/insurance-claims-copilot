"""Things the endpoints need, built on first use and then reused.

Nothing here runs at import time, so the app starts (and the tests run) without an API key or a built index.
Tests replace these functions through FastAPI's dependency_overrides.
"""
from functools import lru_cache

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from fastapi import Depends, HTTPException
from openai import OpenAIError

from .rag.coverage import OpenAICoverageModel
from .rag.embeddings import EMBEDDING_MODEL, OpenAIEmbedder
from .rag.index import PolicyIndexError, check_index, open_client

# Problems that are temporary or on our side of the fence (503). Anything else from upstream is a 502.
UNAVAILABLE_KINDS = ("rate_limited", "timeout", "connection")


def http_error(kind: str, message: object) -> HTTPException:
    """The HTTP error for a failure while talking to the OpenAI API, from an ExtractionError, EmbeddingError or CoverageError."""
    status = 503 if kind in UNAVAILABLE_KINDS else 502
    return HTTPException(status_code=status, detail=f"{kind}: {message}")


@lru_cache
def _index_client() -> ClientAPI:
    return open_client()


def get_index_client() -> ClientAPI:
    return _index_client()


def get_collection(client: ClientAPI = Depends(get_index_client)) -> Collection:
    """The policy index, or a 503 that says what is wrong with it."""
    try:
        return check_index(client, EMBEDDING_MODEL)
    except PolicyIndexError as e:
        raise HTTPException(status_code=503, detail=f"index_{e.kind}: {e}") from e


@lru_cache
def _embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


@lru_cache
def _coverage_model() -> OpenAICoverageModel:
    return OpenAICoverageModel()


def get_embedder() -> OpenAIEmbedder:
    try:
        return _embedder()
    except OpenAIError as e:  # raised when no API key is configured
        raise HTTPException(status_code=503, detail=f"not_configured: {e}") from e


def get_coverage_model() -> OpenAICoverageModel:
    try:
        return _coverage_model()
    except OpenAIError as e:
        raise HTTPException(status_code=503, detail=f"not_configured: {e}") from e
