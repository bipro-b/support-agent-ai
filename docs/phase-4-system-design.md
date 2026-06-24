# Phase 4 — AI System Design

> **Goal of this phase:** turn a capable agent into a deployable *service*. This is the
> "whiteboard" phase — less new ML, more architecture: the API surface, the request
> lifecycle, where state lives, how it scales, and where it breaks. By the end you should
> be able to draw this system on a whiteboard and defend every box, because that is
> exactly what a senior-level system-design interview asks you to do.

Read this, then read `src/support_agent/service/`, then run the API + `phase4_client.py`.

---

## 1. From script to service: what actually changes

Phases 0–3 built the brain. But everything ran as a script: one process, one conversation,
state held in a Python object that lived for the length of the run. A production service is
different in three ways that drive the whole design:

1. **Many concurrent users**, not one. The server handles thousands of conversations.
2. **Stateless transport.** Each HTTP request is independent — the server forgets
   everything between requests unless it deliberately stores it.
3. **It must stay up.** Restarts, multiple replicas, partial failures are normal.

Phase 3's `AgentSession` quietly fails all three. It indexes the entire knowledge base in
its constructor (you can't do that per request), and it bundles per-conversation state
(this customer, this conversation) into the same object as the expensive shared machinery.
Fine for a script; wrong for a service. **The central move of Phase 4 is to split those
two things apart.**

---

## 2. The architecture: three layers

```
        HTTP request
             │
             ▼
   ┌───────────────────────┐
   │  TRANSPORT  (api.py)   │   parse/validate, mint session_id, map errors → status codes
   │  FastAPI               │   THIN — no intelligence here
   └───────────┬───────────┘
               │ handle_turn(customer_id, session_id, message)
               ▼
   ┌───────────────────────┐
   │  DOMAIN  (engine.py)   │   the agent: retrieve → compact → assemble → generate → remember
   │  SupportEngine         │   STATELESS compute, built ONCE, shared by all requests
   │  (LLM, retriever,      │
   │   graph, assembler...) │
   └───────────┬───────────┘
               │ load / save
               ▼
   ┌───────────────────────┐
   │  STATE  (stores)       │   session store (short-term) + memory store (long-term)
   │                        │   + the vector store (knowledge), built offline
   └───────────────────────┘
```

The discipline: **transport knows HTTP but not the agent; the engine knows the agent but
not HTTP; state is just data.** You could put the same engine behind a WebSocket, a queue
consumer, or a CLI without changing a line of agent logic. That separation is what makes a
system testable, swappable, and explainable.

---

## 3. The key idea: a stateless engine + state in stores

`SupportEngine` (`engine.py`) holds only **stateless, shared, read-after-build** machinery:
the LLM client, the retriever (knowledge base indexed **once** at startup), the compiled
graph, the assembler, the summarizer, the extractor. It carries **no per-conversation
data**. That single property is what lets one engine instance safely serve every request
and every thread at the same time — there's nothing per-conversation to corrupt.

Per-conversation state lives in **stores**, loaded and saved each turn:

- **Long-term memory** (`memory_store`) — per-customer facts, keyed by `customer_id`.
- **Short-term session state** (`session_store`) — the conversation, keyed by `session_id`.

So a request doesn't carry a heavyweight object; it carries two ids. The engine loads what
those ids point to, does the turn, saves it back. This is the same statelessness lesson
from Phase 0 (the *model* is stateless; state is resent) lifted one level: now the
*service* is stateless too, and state is reloaded from stores.

> Interview angle: *"Where does conversation state live in your design, and why not in the
> app server?"* → in an external store keyed by session id, so any replica can serve any
> request and a restart doesn't drop the conversation. Holding it in-process couples the
> conversation to one process's memory and lifetime.

---

## 4. The request lifecycle (trace one message end to end)

A POST to `/chat` with `{customer_id, message, session_id?}` runs `handle_turn`:

1. **Transport** validates the body (Pydantic → 422 on garbage), and if there's no
   `session_id` it mints one and returns it (so the client can continue the conversation).
2. **Load state:** `session_store.load(session_id)` → the conversation (or a fresh one);
   `memory_store.get(customer_id)` → the customer's long-term memory.
3. **Retrieve** (Phase 1): pull relevant KB chunks for the message.
4. **Compact** (Phase 2): if history is over budget, summarize old turns.
5. **Assemble** (Phase 2): build the prompt — stable persona+memory prefix, volatile body.
6. **Generate** (Phase 3): run the agent graph — tool loop + human-in-the-loop.
7. **Remember:** extract durable facts into long-term memory.
8. **Save state:** write the conversation back to the session store and the customer back
   to the memory store.
9. **Transport** shapes the result into `ChatResponse` and returns it.

Every phase of this curriculum appears as one step of this single function. That's the
system coming together.

---

## 5. Where state lives (and what it becomes in production)

| State | Keyed by | Lifetime | Our impl | Production |
|-------|----------|----------|----------|------------|
| **Knowledge base** (vectors) | — | Rebuilt on KB change | in-memory numpy, indexed at startup | Vector DB (Qdrant/pgvector), **indexed offline** as a job |
| **Short-term** (conversation) | `session_id` | Minutes–hours | `InMemorySessionStore` (dict) | **Redis** with a TTL (fast, shared across replicas, auto-evicts) |
| **Long-term** (customer facts) | `customer_id` | Durable | JSON files on disk | Postgres (structured) or a vector store (semantic recall) |

Two things to internalize:

- **Each kind of state has different access patterns**, so each gets a different backend.
  Hot, ephemeral session state → Redis. Durable customer facts → a database. Large static
  knowledge → a vector DB indexed by an offline pipeline, *not* on the request path. There
  is no single "the database" for an AI service.
- **Coding to interfaces is what makes this swappable.** `SessionStore` and `MemoryStore`
  are Protocols; `InMemorySessionStore` persists the *serialized* form
  (`ConversationMemory.to_dict()`), exactly as a Redis store would — so dropping in
  `RedisSessionStore` touches nothing else. That's not gold-plating; it's the seam that
  lets dev be cheap and prod be real.

---

## 6. Scaling

Because the engine is stateless and state is external, scaling is the standard web story:

- **Scale horizontally.** Run N identical replicas of the service behind a load balancer.
  Any replica can serve any request *because the conversation isn't trapped in one
  process's memory* — it's in Redis. (This is precisely why we externalized state in §3.
  With the in-memory session store, you'd need **session affinity** — pinning a session to
  one replica — which is fragile and breaks on restart. Externalizing state removes that
  constraint.)
- **Index the knowledge base offline.** Embedding the whole KB is an expensive batch job.
  It runs on a schedule / on KB change and writes to the vector DB; the request path only
  *queries* the index. Never index on the request path.
- **Right-size the model per step.** Extraction and summarization already use the fast
  model; the customer-facing answer uses the strong one. This is the seam Phase 7
  (cost/latency) widens.
- **The bottleneck is almost always the model API**, not your code — latency and rate
  limits. Mitigations: streaming (perceived latency), prompt caching (Phase 2), concurrency
  limits, and queuing. Your service is mostly waiting on the LLM.

---

## 7. Failure domains (where it breaks — full treatment in Phase 8)

A senior design names the failure modes, not just the happy path. Each dependency is a
failure domain:

- **Model API down / rate-limited / slow** → the most likely failure. Needs retries with
  backoff, timeouts, maybe a fallback model, and graceful "try again" messaging.
- **Vector store down** → retrieval fails. Degrade: answer from model + memory without RAG,
  or fail clearly — don't hang.
- **Session store down** → the conversation loses continuity. Degrade to a stateless single
  turn rather than erroring the whole request.
- **A tool/backend errors** → already handled in Phase 3 (the tool returns an error result
  the model can recover from).
- **Bad input / oversized request** → rejected at the Pydantic boundary with a 422.

The theme: **every external call is a thing that can fail, and the design decides what
happens when it does.** Phase 8 builds the actual retries, timeouts, fallbacks, and circuit
breakers. Here we just map the domains — naming them is itself the design skill.

---

## 8. Concurrency correctness

The shared engine is safe under concurrency because its components are read-only after
build (the LLM client, the indexed retriever, the compiled graph hold no per-conversation
mutable state). The one real hazard: **two simultaneous requests for the same session**
(a user double-sends). Both load the same conversation, both save — last write wins, and a
turn can be lost. Production fixes this with a per-session lock or optimistic concurrency in
the session store. Worth *naming* in an interview even when you don't implement it; it
shows you think about correctness under load, not just the single-request path.

---

## 9. Interview-angle checklist

- *Design a customer-support chatbot service.* → three layers (transport/domain/state);
  stateless engine built once; state in external stores keyed by session/customer; the
  request lifecycle of §4.
- *Where does conversation state live and why?* → external store (Redis) keyed by session
  id, so any replica serves any request and restarts don't drop conversations.
- *How do you scale it?* → horizontal replicas of the stateless engine; externalized state
  removes session affinity; index the KB offline; the model API is the bottleneck.
- *What are the failure domains?* → model API, vector store, session store, tools, input;
  name the degradation for each (§7).
- *Why not hold the conversation in the app server's memory?* → couples it to one process's
  lifetime and memory; breaks multi-replica and restarts.
- *Where's the expensive work and how do you keep it off the request path?* → KB indexing →
  offline job; engine built once at startup, not per request.
- *How is this testable without a real model?* → transport depends on the engine via DI;
  override it with a fake in tests (we do exactly this to verify the API).

---

## 10. Exercises (do before Phase 5)

1. **Run the service.** `uvicorn support_agent.service.api:app --reload`, then
   `python examples/phase4_client.py`. Confirm turn 3 ("what order number did I ask
   about?") works — proving server-side session memory.
2. Open `http://127.0.0.1:8000/docs` (FastAPI's auto-generated UI). Send a `/chat` request
   from the browser. Notice the first call is slow (engine + KB indexing built on first
   use) and later calls are fast.
3. **Prove statelessness of the client:** run `phase4_client.py` twice. The second run is a
   brand-new client process with no memory, yet the server still knows Rahim (long-term
   memory on disk) — and starts a fresh session (new session_id).
4. **Swap a backend (conceptually):** read `sessions.py`. Sketch a `RedisSessionStore` with
   the same `load`/`save` using `ConversationMemory.to_dict()/from_dict()`. What else in the
   codebase changes? (Answer: nothing — that's the point of the Protocol.)
5. **Find the affinity problem:** with `InMemorySessionStore`, why would running two server
   replicas behind a load balancer break conversations? Write the two-sentence answer.
6. **Write it down:** in 8 sentences, give the system-design pitch for this service —
   layers, stateless engine, where each kind of state lives, how it scales, top failure
   domain. That's your interview answer.

---

**Next:** Phase 5 — Evals. We can now serve the agent, but "is it any good?" is still a
vibe. We'll build datasets and metrics for answer quality, tool-use correctness, and
groundedness (LLM-as-judge), and make them runnable in CI so a change can't silently
regress quality. Tell me when you're ready.
