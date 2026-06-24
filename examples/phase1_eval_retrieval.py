"""Phase 1 — Measuring retrieval quality.

This is the example that separates "I built a RAG demo" from "I can engineer RAG".
We run the labeled eval set (tests/eval/retrieval_qa.json) through the retriever and
print hit rate, MRR, and recall at several values of k.

If you have a Voyage key, it ALSO compares retrieval WITHOUT reranking vs WITH
reranking, so you can see the second-stage reranker earn its keep (watch MRR — the
reranker pushes the right source higher even when hit rate is already high).

Run:

    python examples/phase1_eval_retrieval.py

Try this experiment: run it once with the hash fallback (no VOYAGE_API_KEY), note the
numbers, then add a Voyage key and run again. The jump is the value of real semantic
embeddings, made measurable.
"""

from __future__ import annotations

from pathlib import Path

from support_agent.rag.evaluation import evaluate_retriever, load_eval_set
from support_agent.rag.retriever import Retriever

EVAL_PATH = Path(__file__).resolve().parents[1] / "tests" / "eval" / "retrieval_qa.json"


def main() -> None:
    retriever = Retriever()
    count = retriever.index_knowledge_base()
    examples = load_eval_set(EVAL_PATH)

    print(f"Indexed {count} chunks | embedder={retriever.embedder.name} "
          f"| reranker={'voyage' if retriever.can_rerank else 'none'}")
    print(f"Evaluating on {len(examples)} labeled questions.\n")

    print("Embedding search only (no rerank):")
    for k in (1, 3, 5):
        print("  " + str(evaluate_retriever(retriever, examples, k=k, rerank=False)))

    if retriever.can_rerank:
        print("\nWith reranking (retrieve top-N, rerank to top-k):")
        for k in (1, 3, 5):
            print("  " + str(evaluate_retriever(retriever, examples, k=k, rerank=True)))
        print("\nCompare the two MRR columns: that lift is the reranker working.")
    else:
        print("\n(No Voyage key -> no reranking stage. Add VOYAGE_API_KEY to compare.)")

    print("\nHow to read this:")
    print("  hit_rate@k = fraction of questions with the right source in the top-k")
    print("  mrr@k      = how HIGH the right source ranks (1.0 = always rank 1)")
    print("  recall@k   = fraction of all gold sources retrieved (matters for multi-doc)")


if __name__ == "__main__":
    main()
