"""An in-memory vector store.

A vector store does two things: hold (vector, chunk) pairs, and given a query
vector, return the most similar chunks. "Most similar" here means highest **cosine
similarity** — the cosine of the angle between two vectors, which is 1.0 when they
point the same way and 0.0 when orthogonal. Because embeddings.py L2-normalizes
every vector, cosine similarity reduces to a plain dot product.

We implement the simplest possible thing: store all vectors in one numpy matrix and,
for each query, compute the dot product against ALL of them and take the top-k. This
is **brute-force / exact** search — O(n) per query.

Why that's fine here, and what production does instead:
- For a few hundred chunks, brute force is instant and exactly correct.
- At millions of vectors, O(n) per query is too slow. Production uses an
  **Approximate Nearest Neighbor (ANN)** index (HNSW, IVF) in a vector database
  (Qdrant, pgvector, Pinecone, ...). ANN trades a tiny bit of recall for massive
  speed. We'll talk about that scaling decision in Phase 4 (system design). The
  *interface* — add vectors, search top-k — is identical, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .documents import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (n_chunks, dim)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("chunks and vectors length mismatch")
        self._chunks.extend(chunks)
        self._matrix = (
            vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        )

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []
        # query_vector is (dim,); matrix is (n, dim). Dot product -> (n,) scores.
        scores = self._matrix @ query_vector
        k = min(k, len(self._chunks))
        # argpartition gets the top-k cheaply; then we sort just those k by score.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [ScoredChunk(self._chunks[i], float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._chunks)
