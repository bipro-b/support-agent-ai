# Phase 2 — Context & Memory Management

> **Goal of this phase:** the model is stateless and its context window is finite. Learn
> to give it the *right* context every turn — conversation history, customer profile,
> retrieved knowledge — without blowing the token budget. This is the discipline that
> separates "a chatbot that works in a demo" from "a chatbot that stays correct, cheap,
> and fast over a 60-turn conversation."

Read this, then read `src/support_agent/memory/`, then run both Phase 2 examples.

---

## 1. The two hard facts this phase exists to manage

Everything here follows from two properties you already met in Phase 0:

1. **The model is stateless.** It remembers nothing between calls. Any "memory" is *you*
   resending context. So memory is a thing *you build and own*, not a feature you turn on.
2. **The context window is finite, and every token in it costs money and latency — every
   turn.** A conversation grows each turn; resent verbatim, it gets linearly more
   expensive and eventually overflows. So you can't just "keep everything."

The job of Phase 2 is to resolve the tension between those two: **hold enough context to
be coherent, but not so much that it's wasteful or overflows.** That tradeoff — what to
keep verbatim, what to compress, what to drop, what to fetch on demand — is context
engineering, and it's a core senior-level skill.

---

## 2. Two kinds of memory

| | Short-term memory | Long-term memory |
|---|---|---|
| **Scope** | One session/conversation | Across all of a customer's sessions |
| **Holds** | The running message list | Durable facts (name, orders, issues, prefs) |
| **Lifetime** | Dies when the session ends | Persists (disk/DB) |
| **File** | `conversation.py` | `store.py` |
| **Analogy** | What you're holding in your head right now | What you know about a regular customer |

Run `phase2_memory.py` **twice**: the first run the agent learns Rahim's name and order;
the second run it already greets him knowing them — because long-term memory was written
to `data/memory/cust_rahim.json` and reloaded. That persistence across process restarts is
the whole point of long-term memory.

---

## 3. Short-term memory: the conversation, and why stored ≠ sent

Short-term memory (`ConversationMemory`) is just the list of user/assistant messages we
keep and resend. Two design decisions matter:

- **The stored history stays clean.** We store exactly what was said — not the persona,
  not retrieved sources, not summaries. Those are *assembled into the prompt* each turn
  and thrown away; only the raw exchange is recorded.
- **A `summary` slot** for compressed older turns (Section 5).

The principle: **the conversation you STORE is not the prompt you SEND.** The prompt is
*built from* stored memory every turn by the assembler (Section 6). This separation is
what lets you change how you build the prompt — summarize, inject RAG, reorder — without
ever corrupting the record of what actually happened.

> Interview angle: *"Where do retrieved documents and summaries live — in the chat
> history?"* No. The history is the clean record; the prompt is reassembled per turn. Mixing
> them rots your memory and makes debugging impossible.

---

## 4. Long-term memory: writing it automatically

Long-term memory (`store.py`) is a small, durable per-customer profile. Two parts:

- **A storage interface + a JSON-file implementation.** `MemoryStore` is a Protocol;
  `JsonFileMemoryStore` writes one JSON file per customer. In production this is a
  database — Postgres for structured facts, Redis for hot session state, or a **vector
  store for *semantic* memory** you retrieve by relevance (when a customer has hundreds of
  facts, you don't dump them all in — you retrieve the relevant few, RAG-style). The
  interface is the contract; swapping the backend shouldn't touch the rest of the system.

- **A `MemoryExtractor`** that reads each turn and pulls durable facts ("the customer's
  name is Rahim", "order #1234 arrived late") using the **fast model** (Haiku). Extraction
  is a simple task — paying Opus for it would be wasteful. This is how memory gets written
  without a human curating it.

Two cautions worth stating now (hardened in Phase 9):

- **Memory is a privacy surface.** Never persist secrets (card numbers, passwords); be
  deliberate about PII. The extractor's system prompt explicitly excludes sensitive data.
- **Memory can be poisoned.** If you blindly store whatever a user says, a malicious user
  can write false "facts" that steer later turns. Real systems validate what they persist.

---

## 5. Summarization: how a long conversation stays in budget

The core technique. When the verbatim history exceeds a token budget, **summarize the
oldest turns into a compact running summary and drop them**, keeping only the recent turns
verbatim (`summarizer.py`, `compact_if_needed`).

```
turns 1..46 (old)        turns 47..50 (recent)
       │                         │
   summarize() ───► summary      └──► kept verbatim
```

The agent still "remembers" the early conversation — just compressed. We:

- keep the last `keep_recent_messages` turns verbatim,
- fold everything older into `conversation.summary` (re-summarizing the prior summary +
  the newly-dropped turns, so it stays bounded too),
- summarize with the **fast model** — again, a cheap task.

This is exactly what the API's server-side **compaction** feature does; we do it by hand so
the mechanics are visible and under our control. In `phase2_memory.py` we set a tiny budget
so you can watch `>>> older turns SUMMARIZED` fire within a few turns.

**A token-counting tradeoff worth internalizing:** deciding "are we over budget?" every
turn with the API's exact counter means a network call each time. For a budget *gate*, a
fast local estimate (`estimate_tokens`, ≈ chars/4) is plenty; reserve the exact counter for
when precision actually matters (e.g., a hard "will this fit?" check before a big call).
**Approximate where it's cheap, exact where it counts.**

Summarization is lossy — that's the deal. The art is choosing what the summary must
preserve (goals, order numbers, commitments, open issues) versus what's safe to drop
(pleasantries). The `_SUMMARY_SYSTEM` prompt encodes exactly that.

---

## 6. Context assembly: the heart of the phase

`ContextAssembler.assemble()` turns stored memory into the prompt, every turn:

```
SYSTEM (stable → cacheable):   persona + standing rules + long-term customer memory
MESSAGES (volatile):           recent verbatim turns
                               + current user turn, augmented with:
                                   - summary of older turns
                                   - retrieved RAG sources for THIS question
                                   - the question
```

The ordering is **deliberate and is the key insight of the phase**:

- **Stable content first, volatile content last.** Persona and long-term memory change
  rarely → they go in the system prefix that we *cache* (Section 7). Retrieved sources and
  the question change every turn → they go *after*, in the messages. If you put the
  question, a timestamp, or the retrieved chunks into the system prefix, you'd invalidate
  the cache on every single turn.
- **We augment a COPY of the last user message**, never the stored one — preserving the
  clean record (Section 3).

This is also where **Phase 1 and Phase 2 unite**: retrieval (RAG) and memory both feed the
assembler, which produces one coherent prompt. The `SupportSession.ask()` flow is:
retrieve → compact → assemble → generate → remember. Hold that five-step shape in mind —
in Phase 3 the inside of `ask()` becomes a LangGraph state machine.

---

## 7. Prompt caching: pay for the stable prefix once

Prompt caching stores a stable prompt prefix server-side and serves it on later calls at
**~10% of the input price**. For an agent that resends a large persona/policy prompt every
turn, this is a major cost and latency win. The mechanics you must know:

- **It's a prefix match.** The cached bytes must be **byte-identical** across calls. One
  changed character anywhere in the prefix invalidates everything after it. This is *why*
  Section 6 puts stable content first — so there's a long, identical prefix to cache.
- **There's a minimum cacheable size** — roughly **4096 tokens for Opus**. Below it,
  caching silently does nothing: no error, just `cache_creation=0, cache_read=0`. People
  trip over this constantly ("I added caching and nothing changed").
- **Cache writes cost ~1.25×, reads ~0.1×.** Break-even is ~2 reads. Worth it when a big
  prefix is reused many times; pointless for a tiny or single-use prefix.

`phase2_prompt_caching.py` shows both faces: a small persona (no cache — below threshold)
and a large inlined policy manual (real cache hit on call 2, with visible savings). The
mechanism is wired in `LLMClient.complete(..., cache_system=True)`, which attaches
`cache_control` to the system block.

> Interview angle: *"You added prompt caching and the bill didn't move. Why?"* → prefix
> below the minimum size; or a silent invalidator in the prefix (a timestamp, a per-request
> ID, non-deterministic JSON ordering, a changing tool set) breaking the byte-match.

---

## 8. Common failure modes to recognize

- **Context overflow / runaway cost** — never compacting; history grows unbounded. Fix:
  the budget + summarization of Section 5.
- **Lost-in-the-middle** — stuffing too much context; the model ignores the middle. Fix:
  keep context tight (fewer, better RAG chunks; summaries instead of full old turns).
- **Cache never hits** — prefix too small, or a volatile value (timestamp, question,
  per-turn retrieved docs) sitting in the supposedly-stable prefix. Fix: Section 6 ordering.
- **Memory poisoning** — persisting unvalidated user claims that steer later turns. Fix:
  validate/scope what you store (Phase 9).
- **Stale or wrong long-term memory** — a fact extracted once is wrong forever. Real
  systems let memory be corrected/expired; ours is intentionally simple here.
- **Summary drops something load-bearing** — the order number gets summarized away and the
  agent can't answer "what was my order number?". Fix: tune what the summary must preserve.

---

## 9. Interview-angle checklist

- *The model is stateless — so how does your agent hold a conversation?* → we store and
  resend history; memory is ours to build.
- *A conversation runs 80 turns. What happens to cost and correctness, and what do you do?*
  → input tokens grow every turn; summarize old turns under a budget, keep recent verbatim.
- *Short-term vs long-term memory — what's the difference and where does each live?*
  → session message list vs durable per-customer store (DB / vector store for semantic recall).
- *How do retrieved docs and summaries relate to the chat history?* → they're assembled into
  the prompt per turn, not stored in history; stored record stays clean.
- *How does prompt caching work and when does it pay off?* → byte-identical stable prefix,
  ~4096-token minimum, ~0.1× reads; worth it for a large reused prefix.
- *You enabled caching and saw no savings — debug it.* → below minimum size or a silent
  invalidator in the prefix.
- *How do you decide what to summarize vs keep?* → token budget gate; keep recent turns and
  load-bearing facts (orders, commitments); compress the rest.

---

## 10. Exercises (do before Phase 3)

1. **Run `phase2_memory.py` twice.** Confirm the second run greets Rahim already knowing
   him. Open `data/memory/cust_rahim.json` and read what got persisted.
2. Watch the **summarization** line fire. Then raise `context_budget_tokens` in the example
   to 2000 and rerun — does it still summarize? Explain why.
3. In the last scripted turn the customer asks "what was my order number again?". Did the
   agent answer correctly *after* older turns were summarized? If yes, the summary preserved
   the order number. Make the summary prompt worse (tell it to be very terse) and see it
   break — that's the lossy-summary failure mode, live.
4. **Run `phase2_prompt_caching.py`.** Confirm: small prefix → no cache, large prefix →
   `cache_read > 0` on call 2. Compute the cost difference between call 1 and call 2.
5. Break caching on purpose: in `LLMClient.complete`, append `str(id(messages))` to the
   system string and rerun the caching demo. Watch the cache stop hitting. Explain why in
   one sentence.
6. **Write it down:** in 6 sentences, explain to a junior dev why "just send the whole
   conversation every time" stops working at scale, and the two levers you use instead
   (summarization + caching).

---

**Next:** Phase 3 — Agentic Orchestration with LangGraph. So far `ask()` is a straight
line. Real support needs *actions* (look up an order, start a return) and *control flow*
(route the question, retry, ask a human). We'll model that as a LangGraph state machine —
where LangGraph is the vehicle and agent design is the lesson. Tell me when you're ready.
