"""Phase 1 — RAG end to end: a grounded answer instead of an invented one.

In Phase 0, when asked about store hours the model just made something up. Here we
RETRIEVE the relevant knowledge-base chunks first and instruct the model to answer
ONLY from them. Same model — but now its answers are grounded in Brecx Store's real
policies, and it says "I don't know" when the answer isn't in the context.

Run (after `pip install -e .` and setting keys in .env):

    python examples/phase1_rag.py

Set VOYAGE_API_KEY in .env for real semantic retrieval. Without it, a local fallback
embedder runs so you can still see the pipeline — but the retrieved chunks will be
much worse (that's the point of the eval in phase1_eval_retrieval.py).
"""

from __future__ import annotations

from support_agent.llm import LLMClient, user
from support_agent.rag.retriever import Retriever, build_grounding_context

# The grounding contract. This system prompt is the heart of trustworthy RAG:
# answer from the provided sources, cite them, and refuse to guess. A huge fraction
# of "the AI hallucinated" incidents are really "we let it answer without grounding
# or didn't tell it to abstain."
GROUNDED_SYSTEM = """You are a customer-support agent for Brecx Store.
Answer the customer's question using ONLY the information in the provided sources.
Rules:
- If the answer is not in the sources, say you don't have that information and offer
  to connect them to a human. Do not guess.
- Be concise (1-3 sentences).
- Cite the source numbers you used, like [Source 2]."""

QUESTIONS = [
    "How long does standard shipping take and is it free?",
    "Can I return something I just didn't like, and will I pay for shipping?",
    "Do you accept cryptocurrency?",
    "What is the meaning of life?",  # not in the KB — should refuse gracefully
]


def answer(llm: LLMClient, retriever: Retriever, question: str) -> None:
    print(f"\n{'=' * 70}\nCustomer: {question}\n{'-' * 70}")

    chunks = retriever.retrieve(question)
    print(f"Retrieved {len(chunks)} chunks "
          f"(embedder={retriever.embedder.name}, reranked={retriever.can_rerank}):")
    for i, sc in enumerate(chunks, start=1):
        print(f"  [{i}] {sc.chunk.source} > {sc.chunk.section}  (score={sc.score:.3f})")

    context = build_grounding_context(chunks)
    prompt = f"Sources:\n{context}\n\nCustomer question: {question}"

    print("\nAgent: ", end="", flush=True)
    for piece in llm.stream([user(prompt)], system=GROUNDED_SYSTEM):
        print(piece, end="", flush=True)
    print()


def main() -> None:
    llm = LLMClient()
    retriever = Retriever()
    count = retriever.index_knowledge_base()
    print(f"Indexed {count} chunks from the knowledge base.")

    for q in QUESTIONS:
        answer(llm, retriever, q)

    print(f"\n{'=' * 70}")
    print("Notice: the last question isn't in the KB — a well-grounded agent refuses")
    print("to invent an answer. Compare this to Phase 0, where it would happily guess.")


if __name__ == "__main__":
    main()
