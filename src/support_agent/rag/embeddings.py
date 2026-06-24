"""Turn text into vectors.

An **embedding** maps a piece of text to a fixed-length vector of numbers such that
texts with similar *meaning* land close together in vector space. That's the whole
trick behind semantic retrieval: embed the user's question, embed every chunk, and
the nearest chunks are the most semantically relevant — even when they share no
keywords ("how do I get my money back?" should find the "Refunds" chunk).

Two providers, one interface:

- `VoyageEmbeddingProvider` — the PRODUCTION path. Voyage is Anthropic's recommended
  embedding provider. Real pretrained semantic model. Needs VOYAGE_API_KEY.
- `HashEmbeddingProvider` — a local, keyless FALLBACK so the pipeline runs out of the
  box. It hashes character n-grams into a vector. It captures crude lexical overlap
  only — NOT meaning. Good enough to see the wiring work; useless for real quality.
  When you run the eval (evaluation.py) you'll watch the metrics jump the moment you
  switch from hash to Voyage. That gap IS the lesson.

A subtle but important detail: good embedding APIs let you tag text as a "document"
or a "query". The same sentence is embedded slightly differently depending on its
role, which improves retrieval. We thread that through the interface.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Literal, Protocol

import numpy as np

from ..config import Settings, get_settings

InputType = Literal["document", "query"]


class EmbeddingProvider(Protocol):
    """The contract every embedder must satisfy."""

    name: str
    dim: int

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of L2-normalized vectors.

        We normalize here so that downstream cosine similarity is just a dot
        product (vector_store.py relies on this).
        """
        ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on empty/degenerate vectors
    return (matrix / norms).astype(np.float32)


class VoyageEmbeddingProvider:
    """Production embeddings via Voyage AI."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set; cannot use Voyage embeddings.")
        import voyageai  # imported lazily so the package is only needed on this path

        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._model = settings.embedding_model
        self.name = f"voyage:{self._model}"
        self.dim = 0  # learned from the first response

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray:
        # Voyage batches internally; for a big KB you'd page this. Fine for our size.
        result = self._client.embed(texts, model=self._model, input_type=input_type)
        matrix = np.array(result.embeddings, dtype=np.float32)
        self.dim = matrix.shape[1]
        return _l2_normalize(matrix)


class HashEmbeddingProvider:
    """Keyless local fallback. Lexical only — NOT semantic. For smoke-testing."""

    def __init__(self, dim: int = 512) -> None:
        self.name = "hash:local-fallback"
        self.dim = dim

    def embed(self, texts: list[str], *, input_type: InputType) -> np.ndarray:
        # input_type is ignored here — a hash embedder has no notion of doc vs query.
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in self._ngrams(text.lower()):
                # Hash each n-gram to a bucket; bag-of-hashed-ngrams as the vector.
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vectors[i, bucket] += 1.0
        return _l2_normalize(vectors)

    @staticmethod
    def _ngrams(text: str, n: int = 4) -> list[str]:
        words = text.split()
        grams = list(words)  # unigrams give word-level lexical overlap
        joined = " ".join(words)
        grams += [joined[j : j + n] for j in range(max(0, len(joined) - n + 1))]
        return grams


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Pick the best available provider: Voyage if keyed, else the hash fallback."""
    settings = settings or get_settings()
    if settings.voyage_api_key:
        return VoyageEmbeddingProvider(settings)
    warnings.warn(
        "VOYAGE_API_KEY not set — using the local hash embedder. Retrieval quality "
        "will be poor; this is only for seeing the pipeline run. Add a Voyage key to "
        "data/.env for real semantic retrieval.",
        stacklevel=2,
    )
    return HashEmbeddingProvider()
