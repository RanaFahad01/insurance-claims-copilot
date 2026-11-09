"""Things the endpoints need, built on first use and then reused.

Nothing here runs at import time, so the app starts (and the tests run) without an API key or a built index.
Tests replace these functions through FastAPI's dependency_overrides.
"""
from functools import lru_cache

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from fastapi import Depends, HTTPException
from openai import OpenAIError

from .agent.documents import DocumentStore, InvoiceModel, OpenAIInvoiceModel
from .agent.loop import OpenAIAgentModel
from .agent.policy_terms import load_policy_terms
from .agent.reviews import ReviewStore
from .agent.toolset import ToolDeps
from .agent.trace import TraceWriter
from .rag.coverage import OpenAICoverageModel
from .rag.embeddings import EMBEDDING_MODEL, Embedder, OpenAIEmbedder
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


# ---- the agent (Phase 4)

@lru_cache
def _agent_model() -> OpenAIAgentModel:
    return OpenAIAgentModel()


@lru_cache
def _invoice_model() -> OpenAIInvoiceModel:
    return OpenAIInvoiceModel()


@lru_cache
def _document_store() -> DocumentStore:
    return DocumentStore()


@lru_cache
def _policy_terms() -> dict:
    return load_policy_terms()


@lru_cache
def _review_store() -> ReviewStore:
    return ReviewStore()


@lru_cache
def _trace_writer() -> TraceWriter:
    return TraceWriter()


def get_agent_model() -> OpenAIAgentModel:
    try:
        return _agent_model()
    except OpenAIError as e:
        raise HTTPException(status_code=503, detail=f"not_configured: {e}") from e


def get_invoice_model() -> OpenAIInvoiceModel:
    try:
        return _invoice_model()
    except OpenAIError as e:
        raise HTTPException(status_code=503, detail=f"not_configured: {e}") from e


def get_document_store() -> DocumentStore:
    return _document_store()


def get_review_store() -> ReviewStore:
    return _review_store()


def get_trace_writer() -> TraceWriter:
    return _trace_writer()


def get_tool_deps(
    # The index comes first, so a missing index fails before anything else is built.
    collection: Collection = Depends(get_collection),
    embedder: Embedder = Depends(get_embedder),
    invoice_model: InvoiceModel = Depends(get_invoice_model),
    documents: DocumentStore = Depends(get_document_store),
    reviews: ReviewStore = Depends(get_review_store),
) -> ToolDeps:
    return ToolDeps(collection=collection, embedder=embedder, documents=documents, invoice_model=invoice_model,
                    terms=_policy_terms(), reviews=reviews)
