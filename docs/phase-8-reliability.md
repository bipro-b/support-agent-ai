# Phase 8 — Reliability

> **Goal of this phase:** make the agent survive the real world. The model API *will*
> rate-limit you, overload, and time out; your vector store, session store, and tools will
> have bad days. Reliability is the deliberate answer to "what happens when each dependency
> fails?" — so a hiccup yields a graceful response, not a 500. By the end you should be able
> to name the failure domains, the mechanism that defends each, and the difference between
> retrying, falling back, and degrading.

Read this, then read `src/support_agent/reliability/`, then run
`examples/phase8_reliability.py` (no API key needed).

---

## 1. The premise: everything fails, so plan for it

In Phase 4 we *named* the failure domains; here we *defend* them. The mental shift is to stop
treating external calls as if they succeed and start treating every one as a thing that can
fail — then decide, per call, what failure means:

| Failure domain | Likely failure | Our defense |
|----------------|----------------|-------------|
| **Model API** | 429 rate limit, 529 overload, 5xx, timeout | SDK retries → model fallback → circuit breaker → degrade |
| **Vector store** | down / slow | catch → answer without RAG (degrade) |
| **Session store** | down | catch → stateless single turn (degrade) |
| **Tool backend** | error / timeout | tool returns an error result (Phase 3); retry if transient |
| **Sensitive action** | duplicate on retry | idempotency |
| **Bad input** | malformed request | rejected at the Pydantic boundary (Phase 4) |

The guiding principle: **a dependency failure should degrade the experience, not end it.** A
slower or simpler answer — or an honest "I'm having trouble, let me get a human" — beats an
error page every time.

---

## 2. Retries with backoff and jitter (`retry.py`)

A **transient** failure might succeed on a second try: a rate limit, a 503, a dropped
connection. The fix is to wait and retry — but three details separate a correct retry from a
harmful one:

- **Exponential backoff** — wait longer each time (0.5s → 1s → 2s). Retrying *instantly*
  hammers a struggling dependency and deepens the outage (a "retry storm").
- **Jitter** — randomize the delay. Without it, a thousand clients that failed at the same
  instant all retry at the same instant, creating synchronized load waves. Jitter spreads
  them out.
- **Retry only what's retryable** — a 400/401 fails identically forever; retrying wastes time
  and money. Classify the error; retry transient, raise permanent. (Our demo proves a
  permanent error is tried exactly once.)

**About the model specifically:** the Anthropic SDK *already* retries 429/5xx/timeouts with
backoff internally — we just configure `max_retries` and `timeout` on the client. So
`retry_call` is for **our** dependencies (a session store, a tool's backend), where we own
the policy. Don't double-wrap the SDK's retries.

---

## 3. Circuit breaker (`circuit_breaker.py`)

Retries handle a blip. But if a dependency is *genuinely down*, retrying every request just
makes every user wait through the full retry sequence before failing — and keeps load on the
dead service so it can't recover. A **circuit breaker** detects sustained failure and trips:

```
CLOSED ──(failures hit threshold)──► OPEN ──(cooldown elapses)──► HALF_OPEN
   ▲                                                                 │
   └──────────────(probe succeeds)──────────────────────────────────┘
                   (probe fails → back to OPEN)
```

- **CLOSED** — normal; count consecutive failures.
- **OPEN** — tripped; reject instantly (fail fast) for the cooldown. Users get an immediate
  graceful answer instead of a slow timeout, and the dependency gets breathing room.
- **HALF_OPEN** — after cooldown, let *one* probe through; success → CLOSED, failure → OPEN.

The win is twofold: **bounded latency** during an outage (fail fast, don't wait), and
**giving the dependency room to recover** (stop the stampede). We put one around the model
API in the engine, shared across turns.

---

## 4. Fallback and graceful degradation (`resilient.py`)

`resilient_generate` is the policy for the call that matters most — the model — and it
composes the primitives:

1. **Try the chosen (routed) model.** The SDK already retried transient errors.
2. **On a still-transient failure, fall back to a second model.** An Opus overload becomes a
   Haiku answer — *a weaker answer beats no answer*. (Phase 7's router picks the model;
   Phase 8's fallback picks a backup when that model is unavailable. Different concerns,
   composed.)
3. **On a permanent error (400/auth), don't fall back** — it'll fail identically. Stop.
4. **If everything fails, return a graceful degraded answer** — a calm "try again / I'll get
   a human." It *never raises*; the customer always gets something civil.

A circuit breaker gates the loop: if the model API has been failing, we skip straight to the
degraded answer instead of making this user wait too. The `TurnResult.degraded` flag (and the
API's `degraded` field) marks these so monitoring can alert and the UI can soften the message.

**Degradation elsewhere in the turn** (`engine.handle_turn`): retrieval is wrapped so a
vector-store failure degrades to "answer without sources" (and the grounding prompt then
nudges the agent toward a human handoff); session load/save and memory extraction are
best-effort — a store failure degrades future continuity but never loses the answer we
already produced. Each degradation sets `degraded=True` and logs a structured error
(observable via Phase 6).

---

## 5. Idempotency (`tools.start_return`)

Retries and graceful re-tries create a hazard for **side effects**: if "start a return"
runs, the response is lost to a timeout, and the client retries — you've now created two
returns, two labels, maybe two refunds. **Idempotency** makes a repeated operation safe: the
second `start_return` for the same (order, item) returns the *same* authorization instead of
creating a duplicate.

This is the reliability counterpart to Phase 3's sensitive-tool gate: that controlled *whether*
an action runs; this controls *that it runs at most once*. Real systems key idempotency on an
explicit token the client sends; our (order_id, item) check is the same idea in miniature.

> Interview angle: *"You retry a request that charges a card. What goes wrong and how do you
> fix it?"* → duplicate charges; make the operation idempotent with an idempotency key so the
> retry is a no-op that returns the original result.

---

## 6. Fail closed vs fail open

A judgment call that signals seniority: when a check itself fails, which way do you err?

- **Fail open / degrade** for read-and-inform paths: retrieval down → answer without sources.
  Losing RAG is better than no answer.
- **Fail closed** for irreversible/sensitive actions: if you can't verify a refund is safe or
  approved, *don't run it*. A skipped refund is recoverable; a wrong one isn't. (This is why
  the Phase 3 approval gate defaults to *requiring* approval, and why idempotency guards the
  action.)

Match the failure direction to the cost of being wrong.

---

## 7. Failure modes / pitfalls

- **Retry storm** — instant or unbounded retries amplify an outage. Fix: backoff + jitter +
  cap + circuit breaker.
- **Retrying permanent errors** — burns time/money on a 400/401. Fix: classify; retry only
  transient.
- **Double-charging on retry** — non-idempotent side effects. Fix: idempotency keys.
- **Cascading failure** — one slow dependency ties up every request until the whole service
  stalls. Fix: timeouts + circuit breaker (fail fast).
- **Silent degradation** — you degrade but never signal it, so nobody notices RAG has been
  down for a day. Fix: set a `degraded` flag, log it, alert on the rate (Phase 6).
- **Fallback that's also down** — falling back to the same failing API. Fix: a fallback on a
  *different* failure axis (different model, cached answer) and a final graceful degrade.
- **Unbounded waits** — no timeout, so a hung connection blocks forever. Fix: set timeouts
  (we configure the SDK's).

---

## 8. Interview-angle checklist

- *What happens when the model API rate-limits / goes down?* → SDK retries with backoff →
  fall back to another model → circuit breaker fails fast → graceful degraded answer; never a
  500.
- *Retry correctly — what are the three things?* → exponential backoff, jitter, retry only
  transient errors.
- *What problem does a circuit breaker solve that retries don't?* → sustained outages: bounded
  latency (fail fast) and recovery room (stop the stampede); plus the half-open probe.
- *You retry an action with a side effect — what breaks and how do you fix it?* → duplicates;
  idempotency keys make the retry a safe no-op.
- *Fail open or fail closed?* → degrade (open) for read/inform paths; fail closed for
  irreversible/sensitive actions; match to cost of error.
- *How do you keep a degraded response from going unnoticed?* → flag + structured log + alert
  on the degradation rate.
- *Where do retries live — your code or the SDK?* → the SDK retries the model; you own retries
  for your other dependencies; don't double-wrap.

---

## 9. Exercises (do before Phase 9)

1. **Run `phase8_reliability.py`** (no key needed). Watch the circuit breaker go
   closed→open→half_open→closed, and the model fallback then degrade.
2. In the retry demo, change `rng` back to the default (real jitter) and `sleep` to
   `time.sleep`; observe the (small) real, varied backoff waits.
3. **Trace a degraded turn:** conceptually, if `self.retriever.retrieve` raised, which
   `TurnResult` fields change? (`degraded=True`, `retrieved=[]`, `context=""`.) Why does the
   answer still come back?
4. Make `start_return` *non*-idempotent (remove the dedupe loop) and call it twice in the demo
   — see two RMAs created. That's the bug idempotency prevents. Restore it.
5. Set `breaker_failure_threshold=2` in config and reason about the tradeoff: trips faster
   (protects sooner) but is more easily tripped by a couple of unlucky errors. What's the
   right value, and what does it depend on?
6. **Write it down:** in 6 sentences, explain the difference between retry, fallback, and
   degrade, and give one example of each from this system.

---

**Next:** Phase 9 — Security. The last engineering phase. A user will try to jailbreak the
agent, extract another customer's data, or smuggle instructions through a retrieved document
(prompt injection). We'll add input/output guardrails, prompt-injection defenses, authorization
on tools, PII handling, and rate limiting — the difference between a demo and something you can
point at the public internet. Tell me when you're ready.
