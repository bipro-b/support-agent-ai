# Phase 6 — Observability

> **Goal of this phase:** when something goes wrong in production — a slow turn, a wrong
> answer, a cost spike — be able to see *why*. Evals (Phase 5) tell you quality dropped;
> observability tells you what actually happened on a specific request and across the fleet.
> By the end you should be able to read a trace, explain the three pillars, and say what you
> would and wouldn't log.

Read this, then read `src/support_agent/observability/`, then run
`examples/phase6_observability.py`.

---

## 1. Why this phase exists

An LLM agent is a black box from the outside: a question goes in, an answer comes out. When
the answer is wrong, slow, or expensive, "look at the code" doesn't help — the behavior
depends on *runtime data* the code doesn't show you: what got retrieved, what prompt was
actually assembled, which tools ran, how many model calls it took, what each cost. Without
observability you're guessing. With it, you can answer the questions you actually ask during
an incident:

- *Why was this turn slow?* → the trace shows which step ate the time.
- *Why did it hallucinate?* → the trace shows exactly what was retrieved and sent (the
  question from Phase 1: "first check what was in front of the model").
- *Why did the bill triple?* → metrics show cost per turn changed after a deploy.
- *What did it do for customer X at 3pm?* → pull the trace + logs by trace id.

This is the difference between "the AI is acting weird ¯\\_(ツ)_/¯" and a root cause.

---

## 2. The three pillars

Standard observability has three complementary signals; we build all three, correlated by a
**trace id**:

| Pillar | Question it answers | Granularity | File |
|--------|---------------------|-------------|------|
| **Traces** | What happened *in this one request*, step by step? | one request | `tracing.py` |
| **Logs** | What discrete events occurred (and their details)? | one event | `logging.py` |
| **Metrics** | How is the system doing *across all requests*? | aggregate | `metrics.py` |

They answer different questions and you need all three: a metric tells you p95 latency rose;
a trace tells you *which step* in a slow request was the culprit; a log line ties a specific
error to a specific trace id. The trace id is the thread that stitches them together.

---

## 3. Traces and spans (the core abstraction)

A **trace** is one request (one customer turn). A **span** is one timed step within it, with
free-form attributes. Spans nest into a tree:

```
chat_turn  [3200ms]  customer_id=cust_rahim session_id=obs_demo
  retrieve  [180ms]  num_chunks=4  sources=['shipping.md', ...]
  compact   [0ms]    summarized=False
  assemble  [1ms]    est_tokens=512
  generate  [2900ms]
    llm.complete_with_tools  [900ms]  input_tokens=1200 output_tokens=40 requested_tools=['lookup_order']
    tool.lookup_order        [2ms]
    llm.complete_with_tools  [1900ms] input_tokens=1400 output_tokens=120 requested_tools=[]
  remember  [600ms]
    llm.complete  [600ms]  model=claude-haiku-4-5  input_tokens=300 output_tokens=25 cost_usd=0.0004
```

Read top to bottom and you see *exactly* where the 3.2 seconds and the dollars went — here,
two model calls in `generate` dominate, and the agent looped once through a tool. That tree
is the single most useful artifact for debugging an LLM app. This is the OpenTelemetry mental
model, built small so the mechanics are visible.

**The clever implementation detail — implicit context propagation.** We do NOT thread a
tracer object through every function. The "current span" lives in a `contextvars.ContextVar`;
any code that opens `with span("x"):` automatically nests under whatever span is active on
this execution context. So the LLM client, the graph nodes, and the engine each instrument
*themselves* with no knowledge of each other, and the tree assembles correctly. This is how
real tracing libraries (OpenTelemetry) avoid polluting every signature with a `tracer`
argument — and it even propagates correctly *through* `langgraph.invoke`, so tool spans
created deep inside the graph still land in the right place.

If no trace is active, `span()` still runs (just unattached) — so instrumentation is
always-on at near-zero cost, and collection only happens inside `start_trace`.

---

## 4. Structured logging

`print("retrieved 4 chunks")` is unsearchable noise at scale. **Structured logs**
(`logging.py`) are JSON records with named fields, so a log system can query them ("all turns
where `cost_usd > 0.05`"). Every record carries the current `trace_id`, so you can pull every
log line for one request and align it with that request's trace. That correlation — log line
↔ trace ↔ metric, all by trace id — is the whole point.

---

## 5. Metrics (across requests)

A trace explains one request; **metrics** (`metrics.py`) describe the whole fleet, and you
alert on them. Two things to internalize:

- **Percentiles, not averages.** The mean hides the users having a bad time. p50 is the
  typical experience; **p95 is the slow tail** that generates complaints. LLM latency is
  high-variance (a turn with two tool calls is far slower than a one-shot answer), so the tail
  matters enormously. We report p50 *and* p95.
- **Cost is a first-class metric for LLM apps.** Unlike a normal web service, every request
  has a direct, variable dollar cost. Track it like latency — it's how you catch a prompt
  change that quietly tripled spend. (Phase 7's optimizations are judged on exactly these
  numbers, with quality held by the Phase 5 evals.)

Each finished trace is rolled up (`summarize_trace`) into a per-turn record — latency, cost,
tokens, llm/tool call counts — which feeds both a structured log line and the metrics
aggregator.

---

## 6. How it's wired in

The engine wraps each turn in `start_trace("chat_turn", ...)` and each step (`retrieve`,
`compact`, `assemble`, `generate`, `remember`) in a `span`. The LLM client wraps every model
call in an `llm.*` span carrying tokens and cost — so model calls show up wherever they
happen (under `generate` for the answer, under `remember` for fact extraction, under
`compact` for summarization). The graph wraps each tool execution in a `tool.*` span. None of
these layers know about each other; the contextvar stitches them into one tree.

Observability is **injected and optional**: `SupportEngine(logger=..., metrics=...)`. The
engine works without them (the spans just aren't logged/aggregated), and the trace tree is
attached to `TurnResult.trace` so the example can render it. In production you wouldn't render
trees to stdout — you'd export spans to a backend (OpenTelemetry/Jaeger) or an LLM-specific
tool (LangSmith, Langfuse) and view them in a UI.

---

## 7. What you should NOT log (privacy — expanded in Phase 9)

Observability is a data-exfiltration and PII surface. The prompts and answers flowing through
this system contain customer data; tracing tools have leaked secrets by capturing full
prompts. The discipline:

- **Log metadata, not content, by default** — counts, ids, costs, latencies, tool names,
  chunk *sources* (not chunk *text*). Notice our spans record `num_chunks` and `sources`, not
  the retrieved text; token *counts*, not the prompt.
- **Never log secrets** (API keys, card numbers, passwords).
- **Sample or redact content deliberately** if you genuinely need it for debugging, behind
  access controls and retention limits.

"Log everything" is a tempting default and a compliance incident waiting to happen.

---

## 8. Failure modes / pitfalls

- **No correlation id** → you have logs and traces but can't connect them to one request.
  Fix: a trace id on every signal (we do).
- **Averages only** → the p95 pain is invisible. Fix: percentiles.
- **Logging full prompts/answers** → PII leak, huge log bills. Fix: metadata-first (§7).
- **Cost untracked** → a prompt change silently triples spend and nobody notices until the
  invoice. Fix: cost is a metric.
- **Over-instrumentation** → a span around every trivial call drowns the signal and adds
  overhead. Fix: span the steps you'd actually inspect (model calls, tools, the major stages).
- **Trace recorded but never exported** → it dies in-process. Fix: ship to a backend (prod);
  we keep the tree on the result for teaching.

---

## 9. Interview-angle checklist

- *How do you debug a wrong/slow/expensive answer in production?* → pull the trace by id;
  read the span tree (what was retrieved, prompt size, tool calls, per-step latency/cost).
- *What are the three pillars and how do they relate?* → traces (one request), logs (events),
  metrics (aggregate), correlated by trace id.
- *What's a span and how does nesting work without passing a tracer everywhere?* → a timed
  step with attributes; implicit context propagation via contextvars.
- *Why p95 and not average latency?* → the mean hides the slow tail that drives complaints;
  LLM latency is high-variance.
- *What do you deliberately NOT log?* → prompt/answer content and secrets by default; metadata
  only; sample/redact under controls.
- *How do you catch a cost regression?* → cost-per-turn as a tracked metric; alert on change.
- *What would you use in production?* → OpenTelemetry/Jaeger or LangSmith/Langfuse rather than
  a homegrown tracer; same concepts.

---

## 10. Exercises (do before Phase 7)

1. **Run `phase6_observability.py`.** Read the three trace trees. For the tool turn, find the
   two `llm.complete_with_tools` spans and the `tool.lookup_order` span between them — that's
   the agent loop, visible.
2. In one trace, identify the single most expensive span (highest `cost_usd`) and the slowest
   step. Are they the same? (Usually the answer-generation model call is both.)
3. Read the JSON `turn_complete` log lines. Confirm each has a `trace_id`. Imagine grepping a
   million of these for `cost_usd > 0.05` — that's why structured.
4. Read the final metrics block. Why might p95 be much larger than p50 here? (Tool turns make
   extra model calls.) Add more turns and watch the percentiles move.
5. **Spot the privacy boundary:** find where the `retrieve` span records `sources` (filenames)
   rather than the chunk text. Why is that the right call? Write the one-sentence reason.
6. **Write it down:** in 6 sentences, explain the three pillars, how the trace id ties them
   together, and one thing you would refuse to put in a log.

---

**Next:** Phase 7 — Cost & Latency. Now that we can *measure* (Phase 5 evals + Phase 6
traces/metrics), we can responsibly optimize: model routing (Haiku vs Opus), prompt caching,
streaming, and trimming context — each change judged against the evals so we cut cost and
latency *without* cutting quality. Tell me when you're ready.
