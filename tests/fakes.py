"""Test doubles shared by the RAG tests. They need no network and no API key."""
import hashlib
import math
import re


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
