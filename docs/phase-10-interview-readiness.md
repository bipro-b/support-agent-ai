# Phase 10 — Interview-Readiness Depth Test

> **Goal of this phase:** prove you can *explain* what you built. A Senior AI Engineer is
> hired on judgment — knowing the tradeoffs, the failure modes, and *why* each decision was
> made. This document drills every phase with the questions a senior interviewer actually
> asks, plus the "explain the tradeoff" follow-ups. You answer out loud first; then check
> what a strong answer hits. The gaps are your study list.

---

## How to use this document

1. **Cover the answer.** Read the question, answer it *out loud* (or write it) before reading
   "Strong answer hits." Recall is the test, not recognition.
2. **Be specific to your system.** You built this — name real choices ("I used a token-bucket
   limiter keyed by customer_id", "I enforce authz in the tool, not the prompt"). Specifics
   are what separate "I read about it" from "I did it."
3. **Always volunteer the tradeoff.** For nearly every answer, the senior-level move is to add
   "…and the tradeoff is X." The follow-ups below train that reflex.
4. **Score yourself** (last section). Anything below "I can explain it cold" goes back to that
   phase's doc.

---

## The 60-second project pitch (memorize this shape)

> "I built a production-grade AI customer-support agent end to end. At its core it's a
> RAG-grounded agent: it retrieves from a knowledge base, holds conversation and per-customer
> memory, and runs as a LangGraph state machine that can call tools — look up an order, start
> a return — with human approval gating the sensitive ones. It's served as a stateless FastAPI
> service with conversation state externalized to a store. Around that core I built the things
> that make it *production*: an eval suite (answer quality, groundedness, tool-use) gated in
> CI; full tracing, structured logs, and cost/latency metrics; model routing and prompt
> caching to cut cost without dropping eval scores; retries, fallback, a circuit breaker and
> graceful degradation for reliability; and security — authorization enforced in code,
> prompt-injection hardening, PII redaction, and rate limiting. The throughline is: the model
> is one box; the engineering is everything around it."

That paragraph hits every phase and signals seniority. Practice it until it's natural.

---

## Phase 0 — Foundations

**Q1. Walk me through what actually happens when you "call an LLM."**
- *Strong answer hits:* a stateless `POST /v1/messages` with system + message history; the
  model is a next-token predictor, so output latency scales with length and streaming is
  possible; you pay per token (input *and* output, priced separately); everything the model
  "knows" about your domain must be in the prompt.

**Q2. The API is stateless — what does that imply for your architecture?**
- *Strong answer hits:* you resend the full history each turn; "memory" is something you build
  and own; statelessness is what lets the service scale horizontally (any replica serves any
  request); you own all state management (Phases 2/4).

**Q3. How do you get deterministic output from Opus 4.8?**
- *Strong answer hits:* NOT `temperature=0` — sampling params are removed on Opus 4.8. You
  steer with prompt design, structured outputs, and lower effort. (Knowing this signals you're
  current.)

**Follow-up — when do you enable thinking, and what does it cost?**
- *Hits:* adaptive thinking for genuinely multi-step problems; it costs tokens + latency, so
  it's off for FAQ lookups and on deliberately for hard reasoning.

---

## Phase 1 — RAG & Retrieval Quality

**Q1. Walk me through your RAG pipeline.**
- *Strong answer hits:* load → chunk → embed → index → retrieve → rerank → generate; name what
  each step decides; generation grounds on retrieved sources with an instruction to refuse when
  the answer isn't present.

**Q2. How do you evaluate *retrieval* (not the final answer)?**
- *Strong answer hits:* a labeled set of (question → which source has the answer); metrics:
  hit-rate@k (did we find it at all?), MRR@k (did we rank it high?), recall@k (did we get all
  pieces?); MRR is what exposes a *ranking* problem hit-rate hides.

**Q3. What's reranking and when is it worth it?**
- *Strong answer hits:* a two-stage pattern — cheap embedding search for recall (top-N), then a
  cross-encoder reranker for precision (top-k); worth it when MRR lags hit-rate; it runs only
  on N candidates so cost is bounded; tradeoff is added latency.

**Follow-up — your RAG is slow/expensive at 10M documents. What changes?**
- *Hits:* brute-force O(n) search → an ANN index (HNSW) in a vector DB; index offline, not on
  the request path; retrieve fewer chunks; the recall/latency tradeoff of ANN.

**Follow-up — how did you choose chunk size?**
- *Hits:* split on semantic boundaries (headings) not a magic number; too big = noise + cost,
  too small = lost context; validated against the retrieval eval.

---

## Phase 2 — Context & Memory

**Q1. The model is stateless — how does your agent hold a conversation, and what breaks at
scale?**
- *Strong answer hits:* store + resend history; input tokens grow every turn so a long chat
  gets linearly more expensive and eventually overflows the context window; fix with
  summarization under a token budget (keep recent turns verbatim, compress older ones).

**Q2. Short-term vs long-term memory — what's the difference and where does each live?**
- *Strong answer hits:* short-term = the session's message list (dies with the session);
  long-term = durable per-customer facts (DB / Redis / a vector store for semantic recall);
  long-term is extracted with the cheap model and persisted.

**Q3. How does prompt caching work and when does it pay off?**
- *Strong answer hits:* a byte-identical stable prefix is cached and served at ~10% input
  price; ~4096-token minimum on Opus; worth it for a large prefix reused across turns/loop
  iterations; you must keep volatile content (question, retrieved docs, timestamps) *after* the
  cached prefix.

**Follow-up — you added caching and the bill didn't move. Why?**
- *Hits:* prefix below the minimum size, or a silent invalidator in the prefix (a timestamp,
  per-request id, unsorted JSON, a changing tool set) breaking the byte-match.

**Follow-up — how do retrieved docs and summaries relate to the stored chat history?**
- *Hits:* they're *assembled into the prompt per turn*, not stored in history; the stored
  record stays clean ("stored ≠ sent"), which is what lets you re-assemble differently each
  turn without corrupting the record.

---

## Phase 3 — Agentic Orchestration

**Q1. What is a tool, and where's the safety boundary?**
- *Strong answer hits:* a function the model *requests* via a `tool_use` block; *your code*
  executes it — the model never runs code. "Model proposes, code disposes" is the entire agent
  safety model.

**Q2. Why does an agent need a loop?**
- *Strong answer hits:* tool results are unknown in advance; each result unlocks the next
  decision (look up order → see items → return the damaged one); loop until the model stops
  calling tools; always cap it (recursion limit) to prevent runaways.

**Q3. How do you stop an agent from taking a dangerous action autonomously?**
- *Strong answer hits:* classify sensitive tools; route them through a human-approval gate
  (HITL); in production use a durable interrupt so a real human approves out-of-band and the
  graph resumes; default to requiring approval for anything irreversible.

**Follow-up — why LangGraph over a plain `while` loop?**
- *Hits:* inspectable state/nodes/edges, conditional routing, persistence/checkpointing,
  durable human-in-the-loop, streaming of steps — an orchestration engine you can observe,
  pause, resume; a `while` loop gives you none of that.

**Follow-up — how did you add orchestration without rewriting the app?**
- *Hits:* overrode one generation hook (`_generate`) on the existing session; memory + RAG
  unchanged — orchestration is a strategy plugged into a stable turn lifecycle.

---

## Phase 4 — AI System Design

**Q1. Design a customer-support chatbot service. (The whiteboard question.)**
- *Strong answer hits:* three layers — transport (HTTP/FastAPI, thin), domain (the stateless
  engine), state (external stores); a stateless engine built once and shared; conversation
  state in an external store keyed by session id; long-term memory keyed by customer; the KB
  indexed offline into a vector DB; the request lifecycle (load → retrieve → assemble →
  generate → remember → save).

**Q2. Where does conversation state live and why not in the app server's memory?**
- *Strong answer hits:* in an external store (Redis) keyed by session id; in-process state
  couples the conversation to one process's memory and lifetime, breaks multi-replica and
  restarts, and forces fragile session affinity.

**Q3. How do you scale it, and what's the bottleneck?**
- *Strong answer hits:* horizontal replicas of the stateless engine behind a load balancer
  (externalized state removes affinity); index the KB offline; the bottleneck is almost always
  the model API (latency + rate limits), mitigated with streaming, caching, concurrency limits.

**Follow-up — name the failure domains.**
- *Hits:* model API, vector store, session store, tools, input — with a degradation plan for
  each (the bridge to Phase 8).

---

## Phase 5 — Evals

**Q1. How do you evaluate an LLM agent?**
- *Strong answer hits:* a labeled dataset; per-dimension checks — deterministic where possible
  (tool-use = a set comparison), LLM-as-judge for subjective quality (answer correctness,
  groundedness, refusal); aggregate to a scorecard; gate in CI.

**Q2. When deterministic check vs LLM-judge?**
- *Strong answer hits:* deterministic for objectively checkable things (tool calls, formats,
  exact values) — free, instant, reliable; LLM-judge only for free-text quality/faithfulness
  you can't `==`.

**Q3. What are the pitfalls of LLM-as-judge?**
- *Strong answer hits:* use a *strong* judge model; the rubric is everything (vague → noisy);
  biases (length, style, position); non-determinism (treat as a dataset-level signal);
  calibrate against human labels; constrain output (structured). A judge that rubber-stamps is
  worse than none.

**Follow-up — how do you stop quality regressions, and how do you know a cost optimization
didn't hurt quality?**
- *Hits:* a CI eval gate with a threshold below baseline; every incident becomes a new case;
  compare eval pass rate before/after any change — that's *why* evals precede optimization.

---

## Phase 6 — Observability

**Q1. Debug a wrong / slow / expensive answer in production. What do you do?**
- *Strong answer hits:* pull the trace by id; read the span tree — what was retrieved, the
  prompt size, which tools ran, per-step latency and cost; the first question for a wrong answer
  is "what was actually in front of the model," which the trace shows.

**Q2. What are the three pillars and how do they relate?**
- *Strong answer hits:* traces (one request), logs (discrete events), metrics (aggregate across
  requests); all correlated by a trace id so you can pivot between them.

**Q3. Why p95 over average latency?**
- *Strong answer hits:* the mean hides the slow tail that drives complaints; LLM latency is
  high-variance (a multi-tool turn is far slower than a one-shot); and cost is a first-class
  metric for LLM apps because each request has a direct dollar cost.

**Follow-up — what do you deliberately NOT log?**
- *Hits:* prompt/answer content and secrets by default (PII/leak surface); log metadata —
  counts, ids, sources, token counts; sample/redact content under access controls if needed.

---

## Phase 7 — Cost & Latency

**Q1. How do you cut LLM cost without hurting quality?**
- *Strong answer hits:* model routing (the big lever), prompt caching, context trimming,
  shorter outputs / lower effort, streaming for perceived latency — *each verified against the
  eval set*. The second half (verification) is the senior part.

**Q2. What's the biggest cost lever and how do you implement it?**
- *Strong answer hits:* model routing; a cheap classifier (on the fast model — never the one
  you're avoiding) picks the smallest capable model per turn; bias misclassifications toward
  the strong model (wrong-cheap hurts quality; wrong-expensive only costs money).

**Q3. Average vs p95 latency — which do you optimize, and does streaming make it faster?**
- *Strong answer hits:* optimize p95 (the tool-heavy tail drives pain); streaming doesn't cut
  total time, it cuts *perceived* latency (time-to-first-token), which is what users feel.

**Follow-up — how do you pick how many chunks to retrieve?**
- *Hits:* tune `top_k` against retrieval-eval hit-rate; fewer tokens until recall starts to
  fall — find the floor with data, not a guess.

---

## Phase 8 — Reliability

**Q1. What happens when the model API rate-limits or goes down?**
- *Strong answer hits:* SDK retries transient errors with backoff → fall back to a second model
  → a circuit breaker fails fast during a sustained outage → graceful degraded answer; never a
  500 or stack trace to the customer.

**Q2. Retry correctly — what are the three things, and what do you NOT retry?**
- *Strong answer hits:* exponential backoff, jitter, and retry only *transient* errors
  (429/5xx/timeout); never retry permanent errors (400/401) — they fail identically and waste
  time/money.

**Q3. What does a circuit breaker solve that retries don't?**
- *Strong answer hits:* sustained outages — it bounds latency (fail fast instead of every user
  waiting through retries) and gives the dependency room to recover; half-open probes for
  recovery.

**Follow-up — you retry an action with a side effect (a refund). What breaks, and the fix?**
- *Hits:* duplicate side effects (double refund); idempotency — key the operation so a retry
  returns the original result instead of acting twice.

**Follow-up — fail open or fail closed?**
- *Hits:* degrade (open) for read/inform paths (retrieval down → answer without sources); fail
  closed for irreversible/sensitive actions; match the failure direction to the cost of error.

---

## Phase 9 — Security

**Q1. What's prompt injection (direct vs indirect) and how do you defend?**
- *Strong answer hits:* text overriding the agent's instructions — from the user (direct) or
  from a retrieved doc / tool result (indirect); defend in layers (treat content as data via a
  preamble, input screening as a *signal*, HITL, and decisively authz in code); state plainly
  that injection is *not fully solved* by any prompt.

**Q2. A prompt injection makes the agent fetch another customer's data. Prevent the leak.**
- *Strong answer hits:* don't rely on the prompt — enforce authorization at the tool boundary
  against the *authenticated principal*, not the id the model passed; the tool refuses
  regardless of what the model was convinced to request; this is deterministic code the model
  can't edit. (This is the single most important security answer.)

**Q3. How do you handle PII, and how do you stop abuse/cost blowups?**
- *Strong answer hits:* redact PII in logs/traces before persisting (the leak surface), don't
  store secrets in memory, the owner sees their own data in answers but not in shared logs;
  per-principal token-bucket rate limiting → 429, Redis-backed across replicas.

**Follow-up — why isn't a good system prompt enough?**
- *Hits:* injection defeats prompts; the code-level controls (authz, HITL, rate limits) are
  what actually hold; the prompt raises the bar, the code makes a successful injection harmless.

---

## Cross-cutting / whole-system questions

**Q1. What's the single throughline of your whole design?**
- *Hits:* the model is one box; everything that makes the product good, safe, cheap, and
  trustworthy is the boxes around it — retrieval, memory, orchestration, evals, observability,
  cost control, reliability, security.

**Q2. The agent gave a wrong answer to a customer. Walk me through your response.**
- *Hits:* pull the trace (Phase 6) → see what was retrieved and sent → is it retrieval
  (Phase 1) or generation? → reproduce as an eval case (Phase 5) so it can't regress → fix →
  confirm the eval passes. A *process*, not a guess.

**Q3. Where would you NOT use an agent / where would you push back?**
- *Hits:* if the task is a single deterministic step, a workflow or one call beats an agent
  (less latency, cost, failure surface); agents earn their cost only for open-ended, multi-step
  tasks where errors are recoverable. Seniority = knowing when *not* to add complexity.

**Q4. What would you build next / what are the weaknesses of this system?**
- *Hits:* (pick honest ones) semantic long-term memory retrieval instead of dumping all facts;
  durable HITL via checkpointer; a real vector DB + offline indexing pipeline; per-session
  concurrency locks; richer eval set from production traffic; output-side guardrails. Naming
  real limitations signals maturity.

---

## Rapid-fire (one-liners — know these cold)

- *Tokens?* The unit you're billed and budgeted in; ~4 chars each; use the API counter, not
  tiktoken.
- *Context window?* Max tokens in+out the model considers at once; a hard budget you manage.
- *Hallucination — first fix?* Ground in retrieved sources + instruct it to refuse when unsure.
- *Hit-rate vs MRR?* Found it at all vs ranked it high.
- *Two-stage retrieval?* Embedding recall (top-N) then rerank for precision (top-k).
- *Why summarize history?* Bound cost/latency and stay under the context window.
- *Cache invalidator?* Any byte change in the prefix (timestamp, id, tool set, model switch).
- *Tool-use safety boundary?* Model proposes, your code disposes.
- *Stateless engine + state in stores?* Lets any replica serve any request.
- *LLM-as-judge risk?* Bias + non-determinism; strong judge, tight rubric, human calibration.
- *p95 not average?* The mean hides the tail users feel.
- *Biggest cost lever?* Model routing.
- *Retry rules?* Backoff + jitter + only transient.
- *Circuit breaker?* Fail fast during sustained outages; let the dependency recover.
- *Idempotency?* Make a retried side-effect safe (no double refund).
- *Stop injection data leaks?* Authorize in code against the authenticated principal.
- *Measure before optimize?* Evals + metrics first, so you know cost cuts didn't hurt quality.

---

## "Tradeoffs I made" — talking points (interviewers love these)

Have 3–4 ready to tell as short stories:

- **Brute-force vector search over a vector DB** — chose simplicity/correctness for the scale I
  had; I know the migration path (HNSW/ANN) and the recall/latency tradeoff for when it grows.
- **A homegrown tracer over OpenTelemetry** — to understand the mechanics; in production I'd
  export to OTel/LangSmith; same concepts (spans, context propagation, trace ids).
- **Routing biased toward the expensive model on uncertainty** — accepted some wasted cost to
  protect quality, because a wrong-cheap answer is worse than a wrong-expensive bill, and the
  eval gate told me where the line was.
- **Heuristic input filter as a signal, not a gate** — I deliberately did NOT make it the
  security boundary because injection defeats blocklists; authz in code is the real control.
- **Summarization is lossy on purpose** — I keep load-bearing facts (order numbers,
  commitments) and compress the rest; the tradeoff is recall of old detail vs token budget.

---

## Red-flag answers to avoid

- "I set `temperature=0` for determinism." (Removed on Opus 4.8 — dates you.)
- "We log everything so we can debug." (PII/secret leak; cost.)
- "The system prompt tells it not to leak data, so we're safe." (Injection defeats prompts;
  enforce in code.)
- "It looked better after my change." (No eval = no evidence.)
- "We retry on every error." (Retrying permanent errors; no backoff = retry storm.)
- "We hold the conversation in memory on the server." (Breaks multi-replica + restarts.)
- "Average latency is fine." (Ignores the p95 tail.)
- "We'd just use a bigger model." (Cost/latency ignorance; routing is the lever.)

---

## Self-assessment scorecard

For each phase, rate yourself: **3** = I can explain it cold *with the tradeoff*; **2** = I get
the idea but fumble specifics; **1** = back to the doc.

| Phase | Topic | Score (1–3) |
|------:|-------|:-----------:|
| 0 | Foundations (tokens, statelessness, cost) | |
| 1 | RAG & retrieval quality (hit-rate/MRR, rerank, ANN) | |
| 2 | Context & memory (summarization, caching) | |
| 3 | Orchestration (tools, loop, HITL, LangGraph) | |
| 4 | System design (stateless engine, state in stores, scaling) | |
| 5 | Evals (deterministic vs judge, CI gate) | |
| 6 | Observability (3 pillars, traces, p95) | |
| 7 | Cost & latency (routing, caching, measure-then-optimize) | |
| 8 | Reliability (retry/backoff, breaker, idempotency, degrade) | |
| 9 | Security (authz-in-code, injection, PII, rate limit) | |

**You're interview-ready when every row is a 3 and you can give the 60-second pitch without
notes.** Any 1s or 2s: reread that phase's doc, run its example, and re-answer its questions.

---

## How to practice

1. **Say it out loud.** Reading ≠ explaining. Record yourself answering five questions; listen
   for hedging and missing tradeoffs.
2. **Whiteboard the system** from memory — the three layers and the request lifecycle — in
   under 5 minutes.
3. **Teach it.** Explain one phase to someone non-technical; if you can make memory or RAG
   click for them, you own it.
4. **Run the examples again** and narrate what each line proves — that's your "I built this"
   evidence.
5. **Mock interview.** Have someone ask from the rapid-fire list at random; aim for crisp,
   tradeoff-aware answers.

---

You built a production-grade AI system from foundations to security, with the engineering
discipline — measurement, observability, reliability, safety — that the title "Senior AI
Engineer" actually means. The code proves you can build it; this document proves you can
defend it. Go get the job.
