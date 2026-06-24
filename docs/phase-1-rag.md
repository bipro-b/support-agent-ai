# Phase 1 — RAG & Retrieval Quality

> **Goal of this phase:** give the agent a real knowledge base so it stops inventing
> answers — and, more importantly, learn to **measure** whether the retrieval is any
> good. Anyone can wire up a RAG demo. A Senior AI Engineer can tell you their hit
> rate, their MRR, why a chunk got missed, and what they'd change.
>
> **বাংলা নোট:** RAG মানে শুধু "vector DB-তে ঢুকিয়ে query করা" না। আসল খেলা হলো
> *retrieval quality* — সঠিক জিনিস উপরে আসছে কিনা, আর সেটা **সংখ্যায়** প্রমাণ করা।
> এই phase-এ আমরা সেটাই শিখব।

Read this, then read the files in `src/support_agent/rag/`, then run both examples.

---

## 1. Why RAG exists

In Phase 0 we asked the model for store hours and it confidently made some up. That's
the core problem: an LLM only knows its frozen training data and whatever is in the
prompt. It does **not** know *your* business — your return policy, your shipping fees,
this customer's order. And when it doesn't know, it often guesses plausibly. That's
hallucination.

**Retrieval-Augmented Generation (RAG)** fixes this by changing *what's in front of
the model*: before answering, we **retrieve** the relevant facts from a knowledge base
and put them in the prompt, then instruct the model to answer only from those facts.

```
question ──▶ [ retrieve relevant chunks ] ──▶ [ stuff them into the prompt ] ──▶ model ──▶ grounded answer
```

The model becomes a *reasoner over provided context* instead of a *recaller of
training data*. That's a smaller, safer, more correct job. Compare `phase1_rag.py`
(grounded) with Phase 0 (invented) — the difference is entirely in what we retrieved.

> **মনে রাখো:** Hallucination-এর বেশিরভাগই আসলে "grounding-এর অভাব"। মডেলকে সঠিক
> context দিলে আর "জানি না হলে বলো" বললে, guessing অনেক কমে যায়।

---

## 2. The RAG pipeline, stage by stage

Every RAG system, however fancy, is these steps. Our files map one-to-one:

| Step | What it does | File |
|------|--------------|------|
| **Load** | Read the knowledge base | `documents.py` |
| **Chunk** | Split docs into small retrievable pieces | `documents.py` |
| **Embed** | Turn each chunk into a vector | `embeddings.py` |
| **Index** | Store vectors for fast search | `vector_store.py` |
| **Retrieve** | Embed the query, find nearest chunks | `retriever.py` |
| **Rerank** | Re-score the top candidates for precision | `retriever.py` |
| **Generate** | Feed chunks + question to the model | `phase1_rag.py` |

The first six are "retrieval". The last is "generation". This phase is almost entirely
about the retrieval half — because that's where RAG quality is won or lost. A perfect
model can't answer correctly from the wrong chunks.

---

## 3. Chunking: the decision that quietly sets your ceiling

You don't retrieve whole documents — you retrieve **chunks**. Why split at all? Because
if a customer asks about the return window, you want the two sentences about the return
window in the prompt, not all of `returns.md`. Tighter context = more accurate answers
and fewer wasted tokens.

Chunking is a **tradeoff with no universally correct answer**:

- **Chunks too big** → each retrieved chunk drags in unrelated text. The model has to
  find the needle in more hay, and you pay for every token. Retrieval also gets fuzzier
  (a big chunk is "about" many things, so its embedding is an average that matches
  weakly).
- **Chunks too small** → context gets severed. A fact that needs its surrounding
  sentence becomes meaningless alone; an answer that spans two chunks may only partly
  get retrieved.

Our strategy (`chunk_documents`): split on the natural `##` section headings — one
topic per chunk — and only fall back to fixed-size overlapping windows if a section is
unusually long. We also **prefix each chunk with its doc title and section heading**
before embedding (`Chunk.for_embedding`), which gives short chunks topical context and
measurably improves retrieval.

**Overlap**: when we do window-split, adjacent windows share some text. This prevents a
fact that lands on a boundary from being cut in half across two chunks.

> Interview angle: *"How did you choose your chunk size?"* The wrong answer is "512,
> it's standard." The right answer names the tradeoff and says "I split on semantic
> boundaries where possible and validated the choice against my retrieval eval."

---

## 4. Embeddings: meaning as geometry

An **embedding** maps text to a vector such that *similar meaning → nearby vectors*.
This is what lets retrieval work on meaning, not keywords: "how do I get my money back?"
has no words in common with "Refunds", yet a good embedder places them close together.

Key points from `embeddings.py`:

- **Semantic vs lexical.** Real embedding models (Voyage, OpenAI, etc.) are pretrained
  to capture meaning. Our `HashEmbeddingProvider` fallback only captures crude word
  overlap — it's *lexical*, not semantic. It exists so the pipeline runs without a
  second API key, and so you can **watch the eval numbers jump** when you switch to
  real embeddings. That gap is the whole lesson made quantitative.
- **document vs query input type.** Good embedding APIs embed the same text slightly
  differently depending on whether it's a stored document or a search query. We thread
  this through (`input_type`), because it's free retrieval quality.
- **Normalization.** We L2-normalize every vector so that cosine similarity becomes a
  plain dot product downstream — a small but important implementation detail.

We default to **Voyage AI** because it's Anthropic's recommended pairing with Claude,
it's cheap (free tier), and its SDK is light. The embedding provider is an interface
(`EmbeddingProvider`), so swapping to a local `sentence-transformers` model or another
API later is a one-class change — the rest of the system doesn't care.

---

## 5. The vector store and similarity search

The store (`vector_store.py`) holds all chunk vectors and, given a query vector,
returns the nearest by **cosine similarity** (1.0 = same direction, 0 = unrelated).
Because vectors are normalized, that's just a dot product.

We use **brute-force exact search**: dot the query against every stored vector, take
the top-k. For a few hundred chunks this is instant and exactly correct.

What changes at scale (a Phase 4 preview, but know it now because it's a guaranteed
interview question):

- Brute force is **O(n)** per query. At millions of vectors that's too slow.
- Production uses an **Approximate Nearest Neighbor (ANN)** index — **HNSW** or IVF —
  inside a vector database (Qdrant, pgvector, Pinecone, Milvus). ANN gives up a sliver
  of recall for enormous speedups.
- The *interface* is identical: add vectors, search top-k. That's deliberate — your
  retrieval code shouldn't care whether the store is brute-force or HNSW.

> Interview angle: *"Your RAG is slow at 10M documents. Why, and what do you change?"*
> → Exact O(n) search; switch to an ANN index (HNSW) in a real vector DB; discuss the
> recall/latency tradeoff and `ef_search` tuning.

---

## 6. Two-stage retrieval: recall then precision (reranking)

This is the most important *quality* idea in the phase.

Embedding search is great at **recall** ("the right chunk is probably in the top 20")
but only okay at **precision** ("this exact one is the single best"). Embedding
distance correlates with relevance but isn't the same thing.

So production RAG uses **two stages** (`retriever.py`):

1. **Recall (cheap, wide):** embedding search pulls the top-N candidates (e.g. 20).
2. **Precision (expensive, narrow):** a **reranker** looks at the query and each
   candidate *together* and scores true relevance far more accurately, then we keep the
   top-k (e.g. 4). The expensive model only runs on N items, not the whole KB.

A reranker is a different kind of model (a cross-encoder): instead of embedding query
and document separately and comparing, it reads them jointly, which is slower but much
sharper. Voyage provides one (`rerank-2.5`).

**Why this shows up in the eval:** reranking often leaves *hit rate* unchanged (the
right doc was already in the top-k) while sharply improving *MRR* (it moves the right
doc from rank 3 to rank 1). Run `phase1_eval_retrieval.py` with a Voyage key and watch
exactly that.

> **বাংলা নোট:** প্রথমে embedding দিয়ে ২০টা সম্ভাব্য চাঙ্ক আনো (recall), তারপর
> reranker দিয়ে সেই ২০টা থেকে সেরা ৪টা বাছো (precision)। দামি model অল্প কিছুর উপর
> চলে, তাই খরচ কম, কিন্তু ranking অনেক ভালো হয়।

---

## 7. Measuring retrieval quality (the heart of this phase)

"It feels better" is not engineering. We need numbers. The harness lives in
`evaluation.py` and the labeled data in `tests/eval/retrieval_qa.json`.

**The setup:** a set of questions, each tagged with the source doc(s) that contain the
correct answer ("gold" sources). Crucially, the questions are **paraphrased** — "how do
I get my money back?" not "refund" — so they test *semantic* retrieval, the thing that
actually matters in production, not keyword matching.

**The three metrics:**

- **Hit Rate @k** — fraction of questions with at least one gold source in the top-k.
  "Did we find it at all?" Your first sanity number.
- **MRR @k** (Mean Reciprocal Rank) — average of 1/(rank of first gold source). Rewards
  ranking the answer **high**. Rank 1 → 1.0, rank 3 → 0.33, miss → 0. This is the metric
  that reveals whether reranking or a better embedder is helping the *ordering*.
- **Recall @k** — for multi-source questions, what fraction of gold sources did we get?
  "Did we collect all the pieces?"

**Why label by source doc, not exact chunk?** Robustness. Change the chunker and exact
chunk IDs change, rotting your labels — but "the answer is in `returns.md`" stays true.
Real systems sometimes label at passage level for sharper signal; doc-level is a
durable, pragmatic choice for learning.

**This is the deliverable that makes you employable in RAG.** With this harness you can
now answer, with evidence: did the new chunking help? Is the reranker worth its latency?
Which questions fail, and why? Without it, you're guessing — and interviewers can tell.

---

## 8. Generation: grounding and refusal

Retrieval done, generation is almost anticlimactic — but the **system prompt** does the
heavy lifting (`GROUNDED_SYSTEM` in `phase1_rag.py`). It instructs the model to:

- answer **only** from the provided sources,
- **cite** the source numbers (traceability — matters for debugging in Phase 6),
- and **refuse** ("I don't have that information, let me get a human") when the answer
  isn't in the context, instead of guessing.

That refusal instruction is doing real safety work. Run `phase1_rag.py` and watch the
"meaning of life" question — a grounded agent declines; an ungrounded one would
cheerfully hallucinate. Most "the AI lied to a customer" incidents are a missing
grounding contract, not a model defect.

---

## 9. Failure modes to recognize (and name in an interview)

- **Retrieval miss** — the right chunk never gets retrieved. Causes: bad chunking,
  weak embedder, query/document phrasing mismatch. Diagnosis: the eval's hit rate.
- **Right chunk, low rank** — it's in the top-20 but buried. Fix: reranking. Diagnosis:
  MRR much lower than hit rate.
- **Lost in the middle** — too many chunks stuffed in; the model ignores ones in the
  middle of a long context. Fix: retrieve fewer, better chunks (rerank to top-k).
- **Grounded but wrong** — retrieval was fine, the model misread the context. Fix:
  prompt, or chunk that confusingly mixed topics.
- **Stale knowledge** — the KB changed but the index didn't. Fix: re-index on update
  (a real pipeline concern — embeddings are computed offline and must be refreshed).

The discipline: when an answer is wrong, **first ask what was retrieved**, then look at
generation. The eval + the printed chunks in `phase1_rag.py` give you exactly that.

---

## 10. Interview-angle checklist

- *Walk me through your RAG pipeline.* → load → chunk → embed → index → retrieve →
  rerank → generate; name what each step decides.
- *How do you evaluate retrieval?* → labeled set; hit rate / MRR / recall@k; explain
  what each catches and why MRR exposes ranking problems.
- *What's reranking and when is it worth it?* → cross-encoder second stage for
  precision; worth it when MRR lags hit rate; costs latency, runs on top-N only.
- *How did you pick chunk size?* → semantic boundaries + validated against the eval;
  name the big/small tradeoff.
- *Why semantic embeddings over keyword search?* → meaning vs lexical overlap;
  paraphrased queries; mention hybrid (combine both) as the production sweet spot.
- *Your RAG is slow / expensive at scale. What changes?* → ANN index (HNSW) in a vector
  DB; retrieve fewer chunks; cache embeddings; precompute offline.
- *The agent hallucinated. Where do you look first?* → what was retrieved (not "the
  model is dumb"); then the grounding/refusal prompt.

---

## 11. Exercises (do before Phase 2)

1. **Run the eval** (`phase1_eval_retrieval.py`) with the hash fallback (no Voyage key).
   Note hit rate / MRR @3. Then add `VOYAGE_API_KEY` to `.env` and rerun. Write down the
   numbers side by side — that's the value of real embeddings, measured.
2. With Voyage on, compare the **no-rerank** vs **rerank** blocks. Which metric moved
   most? Explain why in one sentence.
3. **Break chunking on purpose:** set `SUPPORT_AGENT_CHUNK_MAX_CHARS=120` in `.env` and
   rerun the eval. What happens to the metrics, and why?
4. Add a **hard paraphrased question** to `retrieval_qa.json` (e.g. "the jacket I got is
   the wrong size, can I swap it?") with its gold source, and see if retrieval finds it.
   If it misses, that's a real retrieval bug — investigate via the printed chunks.
5. In `phase1_rag.py`, ask a question whose answer spans **two** docs (e.g. coupons +
   gift cards). Did both sources get retrieved? Check against `recall@k`.
6. **Write it down:** in 6 sentences, explain to a junior dev the difference between
   "we retrieved the right document" (hit rate) and "we ranked it first" (MRR), and why
   a reranker can fix the second without changing the first.

---

**Next:** Phase 2 — Context & Memory Management. RAG gives the agent knowledge; now we
give it *conversation*. The model is stateless and the context window is finite — so how
do we assemble the right context every turn (history + customer profile + retrieved
docs) without blowing the token budget? Tell me when you're ready.
