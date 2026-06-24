# Roadmap — Becoming a Production AI Engineer via One Project

This is the master index. Read it once to see the whole map, then go phase by phase.

> **বাংলা নোট:** এই পুরো journey-টা একটাই project — একটা support agent — ধাপে ধাপে গভীর
> করতে করতে production-grade বানানো। প্রতি ধাপে নতুন একটা production concern যোগ হবে। আগের
> ধাপ না বুঝে পরের ধাপে যাবেন না; প্রতিটা স্তর আগেরটার উপর দাঁড়ানো।

## The mental model

A production LLM application is **not** "call the model and return the text." It's a system with
many moving parts around the model:

```
                ┌─────────────────────────────────────────────────────────┐
                │                     YOUR SYSTEM                          │
   user msg ──▶ │  guardrails → retrieval → memory → orchestration ──┐    │
                │       ▲           ▲          ▲           │          │    │
                │       │           │          │           ▼          │    │
                │   security      RAG       context     LLM call ◀────┘    │
                │       │           │          │           │               │
                │       └───────────┴──────────┴───────────┤               │
                │                                           ▼               │
                │   evals ◀── observability ◀── cost/latency ◀── reliability│
   response ◀── │                                                          │
                └─────────────────────────────────────────────────────────┘
```

The model is the engine. **Everything a Senior AI Engineer is paid for is the system around the engine.**
This roadmap builds that system, one concern at a time.

## The phases

### Phase 0 — Foundations ✅
**Question it answers:** What is actually happening when I "call an LLM"? Tokens, context windows,
the Messages API, statelessness, sampling, streaming, cost. Why prompts are the program.
**Ships:** a typed Claude client wrapper + a runnable demo that shows tokens, streaming, cost, and
multi-turn statelessness.
**Doc:** [`phase-0-foundations.md`](phase-0-foundations.md)

### Phase 1 — RAG & Retrieval Quality
**Question:** How does the agent answer from *our* knowledge base, not its training data — and how
do we know the retrieval is any good?
**Ships:** ingestion → chunking → embeddings → vector store → retrieval → reranking, with a
retrieval-quality eval (hit rate, MRR). The hard part isn't RAG; it's *retrieval quality*.

### Phase 2 — Context & Memory Management
**Question:** The model is stateless and the context window is finite. How do we give it the right
context every turn — conversation history, customer profile, summaries — without blowing the budget?
**Ships:** short-term (conversation) + long-term (per-customer) memory, summarization, context
assembly with a token budget, prompt caching.

### Phase 3 — Agentic Orchestration (LangGraph)
**Question:** Real support needs *actions* (look up order, issue refund) and *control flow* (route,
retry, ask a human). How do we model that reliably instead of one giant prompt?
**Ships:** a LangGraph state machine — router, tool nodes, human-in-the-loop, conditional edges.
LangGraph is the *vehicle*; the lesson is agent design.

### Phase 4 — AI System Design
**Question:** How do all the pieces become one deployable service? API surface, request lifecycle,
where state lives, how it scales, where the failure domains are.
**Ships:** a FastAPI service wrapping the graph, with a clean architecture and a written design doc
(the kind you'd whiteboard in an interview).

### Phase 5 — Evals
**Question:** "It feels better" is not engineering. How do we *measure* answer quality, tool-use
correctness, and catch regressions before shipping?
**Ships:** eval datasets, metrics, LLM-as-judge, a CI-runnable eval suite.

### Phase 6 — Observability
**Question:** It's slow / wrong in production. How do we see *why*? What did it retrieve, what did
it send the model, what did each step cost and take?
**Ships:** structured tracing of every step (spans), logging, token/cost/latency metrics.

### Phase 7 — Cost & Latency
**Question:** It works but it's expensive and slow. How do we make it cheap and fast without making
it dumb?
**Ships:** model routing (Haiku vs Opus), prompt caching, streaming, batching — measured against
the Phase 5 evals so we know quality held.

### Phase 8 — Reliability
**Question:** The model API has an outage / rate-limits us / returns garbage. How does the system
degrade gracefully instead of falling over?
**Ships:** retries with backoff, timeouts, fallbacks, circuit breakers, idempotency.

### Phase 9 — Security
**Question:** A user tries to jailbreak the agent, extract another customer's data, or inject
instructions through a document. How do we defend?
**Ships:** prompt-injection defense, PII handling, authz on tools, rate limiting, output guardrails.

### Phase 10 — Interview-Readiness Depth Test
A document that drills every phase with the questions a senior interviewer actually asks — plus the
"explain the tradeoff" follow-ups. You answer; the gaps show you what to revisit.

## How each phase is structured

- A **doc** in `docs/` — the deep explanation. Read first.
- **Code** in `src/support_agent/` — the production implementation.
- A **runnable example** in `examples/` — see it work, then break it.
- (From Phase 5) **tests/evals** in `tests/`.

## Principles we hold throughout

1. **Production patterns from day one** — typed configs, error handling, no hardcoded secrets.
2. **Measure before optimizing** — evals (Phase 5) come before cost/latency (Phase 7) on purpose.
3. **Understand the layer below** — we don't treat LangGraph or the SDK as magic.
4. **Every choice has a tradeoff** — the docs name them, because interviews probe them.
