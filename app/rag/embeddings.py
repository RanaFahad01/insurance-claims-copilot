"""Turn text into vectors. The rest of the code depends on the small Embedder interface, not on OpenAI."""
from typing import Protocol

from dotenv import load_dotenv
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # texts per API call


class EmbeddingError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in the same order."""
        ...


class OpenAIEmbedder:
    def __init__(self, model: str = EMBEDDING_MODEL, client: OpenAI | None = None):
        self.model = model
        self._client = client if client is not None else OpenAI(timeout=30.0, max_retries=3)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except RateLimitError as e:
                raise EmbeddingError("rate_limited", "Rate limit hit after retries") from e
            except APITimeoutError as e:
                raise EmbeddingError("timeout", "Request timed out after retries") from e
            except APIConnectionError as e:
                raise EmbeddingError("connection", "Could not reach the API") from e
            except APIStatusError as e:
                raise EmbeddingError("api_error", f"API returned status {e.status_code}") from e

            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise EmbeddingError("bad_response", f"Asked for {len(batch)} vectors, got {len(ordered)}")
            vectors.extend(list(item.embedding) for item in ordered)
        return vectors
