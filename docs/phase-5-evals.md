# Phase 5 — Evals

> **Goal of this phase:** stop guessing whether the agent is good. Build datasets and
> metrics that *measure* answer quality, groundedness, and tool-use correctness, use an
> LLM-as-judge for the subjective parts, and turn it into a CI gate so a change can't
> silently regress quality. This is the phase that makes every later optimization safe —
> you cannot responsibly do Phase 7 (cost/latency) without it.

Read this, then read `src/support_agent/eval/`, then run `examples/phase5_eval.py`.

---

## 1. Why evals are the backbone, not a chore

Up to now, "is it better?" has been a vibe. You change a prompt, read three answers, and
feel good. That doesn't scale and it isn't engineering: a tweak that fixes one case
silently breaks five others, and you'd never know. **Evals replace opinion with a number
you can track.**

They unlock three things you can't do without them:

- **Regression safety.** Change the prompt, model, chunking, or retrieval → rerun → did the
  number drop? If yes, you caught it before a customer did.
- **Honest optimization.** Phase 7 will route easy questions to a cheaper model. The only
  way to know quality held is to compare eval scores before and after. *Measure, then
  optimize* — that ordering is why Phase 5 comes before Phase 7.
- **A shared definition of "good."** The dataset and rubrics force you to write down what a
  correct answer actually is. That clarity is half the value.

> Interview angle: *"How do you know your RAG/agent changes are improvements?"* → a labeled
> eval set with per-dimension metrics, run before/after, gated in CI. "It looked better" is
> the wrong answer.

---

## 2. The levels of evaluation

Evals exist at different granularities; you want both:

- **Component evals** — test one piece in isolation. We already built one in Phase 1:
  retrieval quality (hit rate, MRR, recall@k). Cheap, fast, precise about *where* a problem
  is.
- **End-to-end / behavioral evals** — test the whole agent's output, the thing the customer
  experiences. That's this phase. Slower and noisier, but it's what actually matters.

You need both because a green component eval can still produce a bad end-to-end answer (good
retrieval, bad generation) and vice versa. When end-to-end fails, component evals tell you
which stage to look at.

---

## 3. The four dimensions we measure

For a support agent, "good" decomposes into:

| Dimension | Question | How we check it |
|-----------|----------|-----------------|
| **Tool-use correctness** | Did it call the right tools? | **Deterministic** — compare expected vs actual tool calls |
| **Refusal / abstention** | Did it decline to guess on out-of-scope questions? | LLM judge |
| **Answer quality** | Is the answer correct & helpful vs the reference? | LLM judge |
| **Groundedness** | Is every claim supported by the retrieved sources? | LLM judge |

The split between **deterministic** and **model-graded** checks is the key design judgment:

- **Prefer deterministic checks whenever the thing is objectively checkable.** Tool calls
  are structured data — "did `start_return` get called?" is a `set` comparison, not an
  opinion. It's free, instant, and 100% reliable. (This is why Phase 5 added structural
  `tools_called` tracking to the graph rather than parsing answer text.)
- **Use the LLM judge only for the genuinely subjective parts** — free-text correctness,
  faithfulness, whether a refusal was graceful. You can't `==` those.

A case passes only if **all** its applicable checks pass (`CaseResult.passed`). Partial
credit hides regressions.

---

## 4. The dataset is the asset

`tests/eval/answer_quality.json` is a set of labeled `EvalCase`s. Each declares what to
expect so a check can be objective:

- `reference` — the key facts a correct answer must reflect (drives the quality judge).
- `expected_tools` — tools the agent should call (drives the deterministic check).
- `should_refuse` — true for out-of-scope questions it must decline.
- `check_grounded` — whether to run the groundedness judge.

Two things about datasets that matter more than any metric:

1. **Curate from reality.** The best cases come from real customer questions, especially the
   hard, ambiguous, and adversarial ones (note our paraphrased questions and the "40% student
   discount?" trap). A dataset of only easy questions gives a flattering, useless score.
2. **It's a living asset.** Every production bug should become a new eval case so it can
   never silently come back. The dataset is how your system remembers its past mistakes.

> Interview angle: *"Where does your eval data come from?"* → curated from real and
> adversarial queries; every incident becomes a regression case. Not a one-time fixture.

---

## 5. LLM-as-judge: powerful, with sharp edges

Most of "good answer" is free text. The practical way to grade it at scale is to have a
strong model score it against a rubric (`judge.py`). This is standard — and interviewers
probe whether you know its failure modes:

- **Judge with a strong model.** A weak judge gives weak signal. We judge with the primary
  model, never the fast one.
- **The rubric is the whole game.** "Is it good?" produces noise. Each rubric defines what
  every score means and what to penalize (e.g. the quality rubric: "penalize factual errors
  heavily"; the groundedness rubric: "an answer that correctly says it lacks the info is
  fully grounded").
- **Known biases:** LLM judges can prefer longer answers, their own writing style, or the
  first option in a pairwise comparison. Mitigate with tight rubrics and, in production,
  **calibrate against human labels** on a sample to check the judge agrees with people.
- **Non-determinism:** the same answer can score slightly differently run to run. Treat the
  score as a signal over the *dataset*, not gospel per item; watch the aggregate trend, and
  consider averaging multiple runs.
- **Constrain the output.** We force a tiny JSON verdict so it's parseable. Structured
  outputs are the production-robust way to guarantee that shape; our regex-tolerant parse is
  the simple version.

The judge is itself a system you should sanity-check — a judge that rubber-stamps everything
is worse than no judge because it gives false confidence.

---

## 6. From scores to a CI gate

`metrics.py` aggregates results two ways:

- **Per-dimension pass rate** — *where* are we weak? (tool use solid but groundedness shaky?)
- **Overall case pass rate** — the single number a gate thresholds on.

`tests/test_eval_quality.py` turns that into a **regression gate**: it asserts the overall
pass rate is ≥ a threshold and fails the build otherwise. Two pragmatic details that make it
real:

- **It skips when there's no API key**, so ordinary CI stays free; you run the eval
  deliberately (a schedule, or pre-release), because it costs model calls.
- **Set the threshold just below your healthy baseline.** Run the eval a few times, see where
  a good build lands, and set the gate a little under that — so normal judge non-determinism
  doesn't flap the build, but a real regression trips it.

This is how "don't silently regress quality" stops being a hope and becomes mechanical.

---

## 7. How it wires into the agent

The runner is decoupled from *how* answers are produced: you pass a `respond_fn(question) ->
AgentResponse{answer, tools_called, context}`. The example wires in the real `SupportEngine`
(built once, KB indexed once), giving each case a **fresh, isolated customer + session** so
cases can't contaminate each other through long-term memory. Tests wire in a fake responder
+ fake judge to verify the harness itself without spending a cent — the same dependency-seam
discipline from Phase 4.

To make the checks possible, this phase added two structural signals to the agent:
`tools_called` (which tools actually ran, from the graph) and `context` (the sources the
model saw) — both now on `TurnResult`. Evals need ground truth about what the agent *did*,
not just what it *said*.

---

## 8. Failure modes / pitfalls to recognize

- **Eval set too easy or too small** → a flattering score that predicts nothing. Fix: add
  hard, adversarial, real cases.
- **Judge rubber-stamps** → false confidence. Fix: tighten rubric, calibrate against humans,
  spot-check judge verdicts.
- **Flaky gate** → threshold set at the baseline, so non-determinism fails good builds. Fix:
  set it below baseline; average runs.
- **Over-trusting one number** → overall pass rate hides a collapsed dimension. Fix: read the
  per-dimension card too.
- **Grading the wrong thing** → judging answer text when a deterministic check on structured
  output (tool calls) would be exact. Fix: deterministic where possible.
- **Eval/prod skew** → evaluating a code path that isn't what production runs. Fix: run evals
  through the same engine the service uses (we do).

---

## 9. Interview-angle checklist

- *How do you evaluate an LLM agent?* → labeled dataset; per-dimension checks
  (deterministic where possible, LLM judge for subjective); aggregate; CI gate.
- *When deterministic vs LLM-judge?* → deterministic for objectively checkable things (tool
  calls, formats); judge for free-text quality/faithfulness.
- *What are the pitfalls of LLM-as-judge?* → weak-judge, vague rubric, length/style/position
  bias, non-determinism; mitigate with strong judge, tight rubric, human calibration.
- *How do you stop quality regressions?* → eval gate in CI with a threshold; every incident
  becomes a new case.
- *How do you measure hallucination?* → groundedness/faithfulness judge against the retrieved
  sources; an honest "I don't know" counts as grounded.
- *How do you know a cost optimization didn't hurt quality?* → compare eval scores
  before/after; that's why evals precede optimization.

---

## 10. Exercises (do before Phase 6)

1. **Run `phase5_eval.py`** with a Voyage key. Read the scorecard. Which dimension is
   weakest? That's your improvement target.
2. **Cause a regression on purpose:** in `memory/context.py`, weaken the `PERSONA` (delete
   the "do not guess / offer a human" rule), rerun the eval. Watch the refusal dimension
   drop. Restore it. You just saw the gate do its job.
3. **Add a case from a "bug":** invent a question the agent currently gets wrong, add it to
   `answer_quality.json` with the correct `reference`, and confirm it fails — then it's a
   permanent regression guard.
4. **Inspect the judge:** print a judge `Verdict.reason` for one case. Do you agree with it?
   If the judge is wrong, your rubric needs work — fix it and rerun.
5. **Run the gate:** `pytest tests/test_eval_quality.py -v`. Try lowering `THRESHOLD` to see
   it pass and raising it to 0.99 to see it fail — that's the CI mechanism.
6. **Write it down:** in 6 sentences, explain why tool-use is checked deterministically but
   answer quality needs a judge, and name two LLM-judge pitfalls and how you'd mitigate them.

---

**Next:** Phase 6 — Observability. Evals tell you quality dropped; observability tells you
*why* — what was retrieved, what prompt was actually sent, which tools ran, and what each
step cost and took. We'll add structured tracing (spans) across the whole turn so a
production issue is debuggable instead of a mystery. Tell me when you're ready.
