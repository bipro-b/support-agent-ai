"""Measuring retrieval quality — the part most people skip, and the reason this
phase exists.

If you can't measure retrieval, you can't improve it, and you can't tell whether a
change (new chunking, new embedding model, adding a reranker) actually helped or just
*felt* like it did. So we build a tiny eval harness.

The setup: a labeled dataset of questions, each tagged with the source doc(s) that
contain the correct answer (the "gold" / "relevant" sources). We run the retriever on
each question and ask: did the right source show up, and how high?

The three metrics, in plain terms:

- **Hit Rate @k** — over all questions, what fraction had at least one gold source in
  the top-k results? "Did we find it at all (within k)?" Simple, intuitive, the first
  number to look at.

- **MRR @k** (Mean Reciprocal Rank) — for each question, take 1/(rank of the first
  gold source); average over questions. Rewards putting the right answer HIGH, not
  just somewhere in the list. Rank 1 -> 1.0, rank 2 -> 0.5, rank 3 -> 0.33, miss -> 0.
  This is the metric that reveals whether a reranker is helping: it can leave hit rate
  unchanged while pushing the right answer from rank 3 to rank 1.

- **Recall @k** — for questions whose answer spans MULTIPLE gold sources, what
  fraction of those sources did we retrieve? "Did we get all the pieces?"

Why label by SOURCE DOC instead of exact chunk? It's robust: if you change the
chunker, exact chunk IDs change and your labels rot, but "the answer is in returns.md"
stays true. Real systems often label at the passage level for finer signal; doc-level
is a pragmatic, durable choice for learning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .retriever import Retriever


@dataclass(frozen=True)
class EvalExample:
    question: str
    relevant_sources: list[str]  # filenames, e.g. ["returns.md"]


@dataclass
class RetrievalMetrics:
    k: int
    hit_rate: float
    mrr: float
    recall: float
    n: int  # number of questions evaluated

    def __str__(self) -> str:
        return (
            f"@{self.k:<2} | hit_rate={self.hit_rate:.3f}  "
            f"mrr={self.mrr:.3f}  recall={self.recall:.3f}  (n={self.n})"
        )


def load_eval_set(path: str | Path) -> list[EvalExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalExample(d["question"], d["relevant_sources"]) for d in data]


def evaluate_retriever(
    retriever: Retriever,
    examples: list[EvalExample],
    *,
    k: int,
    rerank: bool = True,
) -> RetrievalMetrics:
    """Run the retriever on every question and aggregate the three metrics."""
    hits = 0
    reciprocal_ranks = 0.0
    recall_sum = 0.0

    for ex in examples:
        results = retriever.retrieve(ex.question, k=k, rerank=rerank)
        retrieved_sources = [sc.chunk.source for sc in results]
        gold = set(ex.relevant_sources)

        # Hit rate: any gold source anywhere in the top-k?
        if gold.intersection(retrieved_sources):
            hits += 1

        # MRR: rank of the FIRST gold source (1-indexed); 0 if none found.
        rr = 0.0
        for rank, src in enumerate(retrieved_sources, start=1):
            if src in gold:
                rr = 1.0 / rank
                break
        reciprocal_ranks += rr

        # Recall: fraction of distinct gold sources we managed to retrieve.
        found = gold.intersection(retrieved_sources)
        recall_sum += len(found) / len(gold)

    n = len(examples)
    return RetrievalMetrics(
        k=k,
        hit_rate=hits / n,
        mrr=reciprocal_ranks / n,
        recall=recall_sum / n,
        n=n,
    )
