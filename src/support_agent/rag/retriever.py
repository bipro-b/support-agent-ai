"""The retriever: the thing the rest of the app calls to get relevant context.

It owns the two-stage retrieval pattern that production RAG systems use:

  STAGE 1 — RECALL (cheap, wide net):
    Embed the query, search the vector store, pull the top-N (e.g. 20) candidates.
    Embedding search is fast but approximate about *relevance* — it's good at "these
    20 are in the right neighborhood", weaker at "this exact one is best".

  STAGE 2 — PRECISION (expensive, narrow):
    Run a **reranker** over those N candidates. A reranker is a model that looks at
    the query and each candidate *together* and scores true relevance much more
    accurately than embedding distance. Keep the top-k (e.g. 4). We only rerank N
    items, not the whole KB, so the expensive model runs on a small set.

Why bother with two stages? Embedding similarity and actual relevance are correlated
but not the same. The reranker fixes the ordering. In the eval (evaluation.py) you'll
see reranking lift the metrics — that's the payoff, and it's a classic
retrieval-quality lever interviewers ask about.

Reranking needs Voyage. With the hash fallback we skip stage 2 (and say so).
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .documents import Chunk, chunk_documents, load_knowledge_base
from .embeddings import EmbeddingProvider, get_embedding_provider
from .vector_store import InMemoryVectorStore, ScoredChunk


class Retriever:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedding_provider(self.settings)
        self.store = InMemoryVectorStore()
        self._reranker = _maybe_build_reranker(self.settings)

    @property
    def can_rerank(self) -> bool:
        return self._reranker is not None

    # ------------------------------------------------------------------ #
    # Indexing                                                           #
    # ------------------------------------------------------------------ #
    def index_knowledge_base(self) -> int:
        """Load the KB, chunk it, embed it, and fill the vector store.

        Returns the number of chunks indexed. In production this is an offline job,
        not something you do on every request — embeddings are precomputed and
        persisted. We do it in-process here for simplicity.
        """
        docs = load_knowledge_base(self.settings.knowledge_base_dir)
        chunks = chunk_documents(
            docs,
            max_chars=self.settings.chunk_max_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        vectors = self.embedder.embed(
            [c.for_embedding() for c in chunks], input_type="document"
        )
        self.store.add(chunks, vectors)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Retrieval                                                          #
    # ------------------------------------------------------------------ #
    def retrieve(
        self, query: str, *, k: int | None = None, rerank: bool = True
    ) -> list[ScoredChunk]:
        k = k or self.settings.final_top_k
        # Stage 1: wide recall via embedding search.
        n = max(self.settings.retrieve_top_n, k)
        query_vec = self.embedder.embed([query], input_type="query")[0]
        candidates = self.store.search(query_vec, k=n)

        # Stage 2: precision via reranking (if available and requested).
        if rerank and self._reranker is not None and candidates:
            return self._reranker.rerank(query, candidates, top_k=k)
        return candidates[:k]


class _VoyageReranker:
    def __init__(self, settings: Settings) -> None:
        import voyageai

        self._client = voyageai.Client(api_key=settings.voyage_api_key)
        self._model = settings.rerank_model

    def rerank(
        self, query: str, candidates: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        documents = [c.chunk.text for c in candidates]
        result = self._client.rerank(
            query, documents, model=self._model, top_k=top_k
        )
        # Map reranked results back to the original chunks, carrying the new score.
        out: list[ScoredChunk] = []
        for item in result.results:
            original = candidates[item.index]
            out.append(ScoredChunk(chunk=original.chunk, score=item.relevance_score))
        return out


def _maybe_build_reranker(settings: Settings) -> _VoyageReranker | None:
    if settings.voyage_api_key:
        return _VoyageReranker(settings)
    return None


def build_grounding_context(chunks: list[ScoredChunk]) -> str:
    """Format retrieved chunks into a context block to put in the prompt.

    We number the sources so the model can cite them and so a human debugging the
    answer can trace which chunk produced which claim (matters in Phase 6).
    """
    parts: list[str] = []
    for i, sc in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {sc.chunk.source} > {sc.chunk.section}]\n{sc.chunk.text}"
        )
    return "\n\n".join(parts)
