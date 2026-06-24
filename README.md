# Support Agent — A Production-Grade AI Customer Support System

> A single, realistic project built to learn **production AI engineering** end to end —
> not tutorial-level toys. Each phase ships working code **and** a deep written doc you
> can read to actually understand *why*, not just *how*.

## What we're building

A customer-support agent for a fictional e-commerce company (**"BIP Store"**). It:

- Answers customer questions from a knowledge base (**RAG**)
- Remembers the conversation and the customer's history (**context & memory**)
- Takes real actions — look up an order, start a return, escalate to a human (**tools / agentic**)
- Is orchestrated as a state machine with **LangGraph**
- Is measured (**evals**), observed (**observability**), and tuned for **cost / latency**
- Survives failure (**reliability**) and resists abuse (**security**)

This one domain naturally forces us through every topic a Senior AI Engineer is expected to know.

## The learning path

See **[`docs/00-roadmap.md`](docs/00-roadmap.md)** — the master index. Each phase has its own doc.

| Phase | Topic | Doc |
|------:|-------|-----|
| 0 | Foundations — how LLM systems actually work | [`phase-0-foundations.md`](docs/phase-0-foundations.md) ✅ |
| 1 | RAG & retrieval quality | [`phase-1-rag.md`](docs/phase-1-rag.md) ✅ |
| 2 | Context & memory management | [`phase-2-context-memory.md`](docs/phase-2-context-memory.md) ✅ |
| 3 | Agentic orchestration (LangGraph) | [`phase-3-orchestration.md`](docs/phase-3-orchestration.md) ✅ |
| 4 | AI system design | [`phase-4-system-design.md`](docs/phase-4-system-design.md) ✅ |
| 5 | Evals | [`phase-5-evals.md`](docs/phase-5-evals.md) ✅ |
| 6 | Observability | [`phase-6-observability.md`](docs/phase-6-observability.md) ✅ |
| 7 | Cost & latency | [`phase-7-cost-latency.md`](docs/phase-7-cost-latency.md) ✅ |
| 8 | Reliability | [`phase-8-reliability.md`](docs/phase-8-reliability.md) ✅ |
| 9 | Security | [`phase-9-security.md`](docs/phase-9-security.md) ✅ |
| 10 | Interview-readiness depth test | [`phase-10-interview-readiness.md`](docs/phase-10-interview-readiness.md) ✅ |

**All 10 phases complete.** The agent is built foundations-to-security, every phase has a deep
doc and a runnable example, and the capstone is an interview depth-test in
[`docs/phase-10-interview-readiness.md`](docs/phase-10-interview-readiness.md).

## Runnable examples

```
examples/phase0_hello_claude.py      foundations: tokens, streaming, statelessness, cost
examples/phase1_rag.py               grounded answers from the knowledge base
examples/phase1_eval_retrieval.py    retrieval quality: hit-rate / MRR / recall
examples/phase2_memory.py            multi-turn memory + summarization (run twice)
examples/phase2_prompt_caching.py    prompt caching: below vs above the threshold
examples/phase3_agent.py             tools + routing + human-in-the-loop
examples/phase4_client.py            talk to the agent over HTTP (start the server first)
examples/phase5_eval.py              end-to-end eval scorecard (quality/grounded/tools)
examples/phase6_observability.py     trace trees + structured logs + metrics
examples/phase7_cost_latency.py      routing OFF vs ON — measured cost/latency
examples/phase8_reliability.py       retry, circuit breaker, fallback, idempotency  (no key)
examples/phase9_security.py          authz, injection screening, PII, rate limit     (no key)
```

## Stack

- **Python 3.11+**
- **Anthropic Claude** (`claude-opus-4-8`, `claude-haiku-4-5`) as the LLM
- **LangGraph** for orchestration (added in Phase 3)
- A vector store for RAG (added in Phase 1)
- `pytest` for evals, OpenTelemetry-style tracing for observability (later phases)

## Setup

```powershell
# 1. Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -e .

# 3. Configure your API key
Copy-Item .env.example .env
# then edit .env and paste your ANTHROPIC_API_KEY

# 4. Run the Phase 0 demo
python examples/phase0_hello_claude.py
```

Get an API key from https://console.anthropic.com → Settings → API Keys.

## How to use this repo to learn

1. Read the phase's doc in `docs/` **first** — build the mental model.
2. Read the code it refers to in `src/` and `examples/`.
3. Run the example, break it, change parameters, watch what happens.
4. When stuck, ask. Then move to the next phase.
