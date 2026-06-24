# Phase 7 — Cost & Latency

> **Goal of this phase:** make the agent cheaper and faster **without making it dumber**.
> The reason this comes *after* evals (Phase 5) and observability (Phase 6) is not
> accidental: optimization is only responsible when you can measure both what you're saving
> (cost/latency — Phase 6) and what you might be breaking (quality — Phase 5). Optimizing
> blind is how you ship a 60%-cheaper agent that quietly got 20% worse.

Read this, then read `src/support_agent/optimization/`, then run
`examples/phase7_cost_latency.py` followed by the eval gate.

---

## 1. The optimization loop

Every change in this phase follows the same disciplined loop:

```
   measure baseline  ──►  apply one optimization  ──►  measure again
   (cost, latency,         (routing, caching,           (did cost/latency improve?
    eval pass rate)         trimming, ...)                did the eval pass rate hold?)
```

Two numbers move in tension: **cost/latency** (down is good) and **quality** (must not
drop). A change is only a win if it improves the first while holding the second. "It's
cheaper" is trivially easy — send everything to the smallest model and you've cut cost 80%
and quality with it. The skill is cheaper *and still correct*, proven with the eval gate.

> Interview angle: *"How do you reduce LLM cost without hurting quality?"* → name the levers
> (routing, caching, trimming, model/effort tuning), then immediately: "and I verify each
> against an eval set so I know quality held." The second half is what separates senior from
> junior.

---

## 2. Where the cost and latency actually come from

From Phase 6's traces you can see it directly: **the model calls dominate both.** So the
levers all target the model calls:

- **Which model** runs each call (5× price difference between Opus and Haiku).
- **How many tokens** go in (context size) and out (answer length).
- **How many calls** happen (the agent loop; cache hits that skip recomputation).
- **How the user perceives** the wait (streaming changes perceived latency without changing
  total time).

We pull each of these levers.

---

## 3. Lever 1 — Model routing (the big one)

**Most support turns are easy** ("what are your hours?") and a cheap model answers them
perfectly. **A few are hard** (multi-step reasoning, account actions) and need the strong
model. Sending *every* turn to Opus is paying first-class fare for every passenger. A
**router** (`optimization/router.py`) classifies each turn and picks the cheapest model that
can do the job. Because Haiku is ~5× cheaper than Opus, and easy turns are the majority, this
is the single biggest cost lever.

Design points that matter:

- **The router must be cheap.** A router that costs as much as what it saves is pointless. Our
  LLM classifier runs on the *fast* model — never the one we're trying to avoid. (A heuristic
  classifier is also provided: free, zero-latency, but cruder.)
- **Bias toward the safe direction.** A misclassification has two costs: routing a hard turn
  to Haiku *hurts quality*; routing an easy turn to Opus only *wastes money*. The first is
  worse, so on any uncertainty we default to COMPLEX (Opus). You'd rather overspend than be
  wrong.
- **The router is itself in the trace.** A `route` span wraps the decision, and the
  classifier's own model call shows up as a child `llm.*` span — so its cost is counted, not
  hidden. Honest accounting: the router's overhead is part of the measured cost.

**Heuristic brittleness (a real lesson):** the keyword heuristic flags anything containing
"return"/"refund"/"order" as COMPLEX — so the FAQ "what's your *return* window?" gets
over-routed to Opus. That's the safe-but-wasteful direction, and it's exactly why the LLM
classifier (which understands "return window" is a policy question) is the default. Cheap
heuristics trade accuracy for zero cost; know when that trade is worth it.

The chosen model threads through the whole agent graph (`AgentState.model` →
`complete_with_tools(model=...)`), so the entire tool loop for that turn runs on the routed
model, and the reported `usage.model` reflects it.

---

## 4. Lever 2 — Prompt caching in the agent loop

We built prompt caching in Phase 2 (`cache_system`); Phase 7 *applies* it where it pays most:
the agent loop. Each loop iteration resends the same large, stable prefix (persona + customer
memory + tool guidance) with only the messages growing. With caching on, that prefix is
served at ~10% price on iteration 2+ of a turn, and across turns whenever it's unchanged. So
we set `cache_system=True` in the agent node. It's a near-free win: same quality, lower cost
and latency, no behavior change. (Recall the caveats from Phase 2: byte-identical prefix,
~4096-token minimum — our persona + memory + tool guidance is sizable, so it qualifies.)

---

## 5. Lever 3 — Context trimming (fewer tokens in)

Every token in the prompt costs money and adds latency, *every turn*. Two knobs already in
the system:

- **Retrieve fewer, better chunks** (`final_top_k`, Phase 1). Reranking lets you send the top
  4 instead of the top 20 — far fewer input tokens with little quality loss, *if* your
  retrieval eval (Phase 1) confirms the answer is usually in the top few. Lower `final_top_k`
  → cheaper/faster; watch hit-rate@k to find the floor.
- **Summarize aggressively** (`context_budget_tokens`, Phase 2). A tighter budget compresses
  history sooner — cheaper turns, at the risk of dropping detail. The summary's job is to keep
  the load-bearing facts.

Both are tradeoffs you tune *against the evals*, not by feel. This is where Phase 1's
retrieval eval and Phase 5's answer eval pay off as tuning instruments.

---

## 6. Lever 4 — Output length and effort

Output tokens cost ~5× input tokens, so a rambling answer is the expensive kind. Levers:

- **A concise instruction** in the persona (we already ask for 1–3 sentences) directly cuts
  output cost.
- **`max_tokens`** caps the worst case.
- **Effort** (`output_config: {effort: ...}`) controls how much the model thinks/works — lower
  effort on simple turns spends fewer tokens. (We route by *model*; effort is the finer knob
  you'd add next, per-route.)

---

## 7. Lever 5 — Streaming (perceived latency)

Streaming (`LLMClient.stream`, Phase 0) doesn't make a turn finish sooner — it makes it *feel*
sooner by showing the first tokens in ~300ms instead of after the full answer. For a
user-facing chat, **time-to-first-token is the latency users actually feel**, so streaming is
a big perceived-latency win for free. (The engine returns full results so it can do memory +
metrics + tool loops; the API layer is where you'd stream tokens to the client.)

---

## 8. Putting it together: the measured result

`phase7_cost_latency.py` runs a realistic mix of turns twice — routing OFF (all Opus,
baseline) and routing ON — and compares cost and latency from the Phase 6 metrics, printing
which model the router chose per turn. You'll see the easy FAQs go to Haiku and the
order/return turns stay on Opus, with a real cost drop.

Then — and this is the non-negotiable second half — run `pytest tests/test_eval_quality.py`.
If the pass rate held, the optimization is a genuine win. If it dropped, you routed too
aggressively and the gate caught it. **That loop — optimize, then prove quality held — is the
entire point of the phase.**

---

## 9. Failure modes / pitfalls

- **Optimizing without an eval gate** → you ship a cheaper, worse agent and find out from
  customers. Fix: always re-run the evals.
- **Router too aggressive** → quality drops as hard turns go to the weak model. Fix: bias
  toward the strong model on uncertainty; tune against evals.
- **Router too expensive** → classification eats the savings. Fix: classify with the fast
  model (or a heuristic).
- **Caching that never hits** → a volatile value in the "stable" prefix (timestamp, per-turn
  data). Fix: keep volatile content out of the cached prefix (Phase 2).
- **Trimming below the recall floor** → fewer chunks but now the answer isn't in them. Fix:
  set `final_top_k` using the retrieval eval, not by guessing.
- **Chasing average latency** → the p95 tail (multi-tool turns) is what hurts. Fix: optimize
  the tail; measure p95 (Phase 6).
- **Premature optimization** → tuning before you have evals/metrics. Fix: that's why this
  phase is #7, not #2.

---

## 10. Interview-angle checklist

- *How do you cut LLM cost without hurting quality?* → routing, caching, context trimming,
  output/effort tuning, streaming for perceived latency — each verified against evals.
- *What's the biggest cost lever and how do you implement it?* → model routing; a cheap
  classifier picks the smallest capable model per request; bias to the strong model on doubt.
- *How do you keep the router from eating its own savings?* → classify with the fast model or
  a heuristic; it must be far cheaper than what it avoids.
- *How do you decide how many chunks to retrieve?* → tune `top_k` against retrieval-eval
  hit-rate; fewer tokens until recall starts to fall.
- *Average vs p95 latency — which do you optimize?* → p95; the tail (tool-heavy turns) drives
  user pain; the mean hides it.
- *Streaming — does it make things faster?* → not total time; it cuts *perceived* latency
  (time-to-first-token), which is what users feel.
- *How do you know an optimization was safe?* → the eval pass rate held; cheaper-and-correct,
  proven, not assumed.

---

## 11. Exercises (do before Phase 8)

1. **Run `phase7_cost_latency.py`.** Read the per-turn model choices: which turns went to
   Haiku, which stayed on Opus? Does the split match your intuition about difficulty?
2. Note the baseline vs routed cost. What's the % saving on this mix? Why isn't it the full
   80% (hint: the complex turns still use Opus, and the router itself costs a little)?
3. **Prove quality held:** run `pytest tests/test_eval_quality.py` with routing on. Did the
   pass rate stay above threshold? That's the whole game.
4. **Route too hard:** in `router.py`, make `_classify_llm` always return SIMPLE (everything →
   Haiku). Rerun the eval. Watch the pass rate drop on the tool/complex cases — the gate
   catching an over-aggressive optimization.
5. **Tune trimming:** lower `final_top_k` to 2 in config, run the Phase 1 retrieval eval and
   the Phase 5 eval. Cheaper — but did recall/quality fall? Find the floor.
6. **Write it down:** in 6 sentences, explain why model routing is the biggest cost lever,
   which direction you bias misclassifications and why, and how you prove the optimization
   didn't cost you quality.

---

**Next:** Phase 8 — Reliability. The model API will rate-limit you, time out, and occasionally
return errors; your tools and stores will fail. We'll add retries with backoff, timeouts,
fallbacks, and graceful degradation so a dependency hiccup doesn't take down the whole turn.
Tell me when you're ready.
