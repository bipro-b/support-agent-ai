"""Retrieval-Augmented Generation (Phase 1).

Pipeline:
    documents.py    load the knowledge base, split into chunks
    embeddings.py   turn text into vectors (Voyage in prod, hash fallback locally)
    vector_store.py hold vectors, find the nearest by cosine similarity
    retriever.py    tie it together: embed query -> search -> (optional) rerank
    evaluation.py   MEASURE retrieval quality: hit rate, MRR, recall@k

Read docs/phase-1-rag.md alongside these files.
"""
