"""SupportEngine — the stateless, shared core of the service.

This is the heart of the Phase 4 design. Phase 3's `AgentSession` was a convenient object
that bundled together the expensive shared machinery (LLM client, the indexed retriever,
the compiled graph) AND the per-conversation state (this customer, this conversation).
That's fine for a script — one object, one conversation. It's wrong for a service:

  - The retriever indexes the whole knowledge base on construction. Doing that per request
    (or per session object) is absurd — it must happen ONCE at startup.
  - A web server handles many customers concurrently. You can't have one giant object per
    live conversation sitting in memory forever.

So we split:

  SupportEngine  = the STATELESS shared compute. Built ONCE. Holds no per-conversation
                   data — every turn loads the state it needs from stores and saves it
                   back. Because it carries no mutable per-conversation state, a single
                   engine instance can safely serve every request/thread.

  stores         = WHERE STATE LIVES. Long-term memory (per customer) and session state
                   (per conversation) are loaded and saved per request.

`handle_turn` is the full request lifecycle for one message:
    load state -> retrieve -> compact -> assemble -> generate (graph) -> remember -> save
"""

from __future__ import annotations

from ..agent.agent_session import TOOL_GUIDANCE, auto_approve
from ..agent.graph import ApprovalCallback, build_support_graph, run_agent_turn
from ..config import Settings, get_settings
from ..llm import LLMClient
from ..memory.context import ContextAssembler
from ..memory.conversation import ConversationMemory
from ..memory.session import TurnResult
from ..memory.store import JsonFileMemoryStore, MemoryExtractor, MemoryStore
from ..memory.summarizer import Summarizer, compact_if_needed
from ..observability.logging import StructuredLogger
from ..observability.metrics import MetricsAggregator
from ..observability.tracing import span, start_trace, summarize_trace
from ..optimization.router import ModelRouter
from ..rag.retriever import Retriever, build_grounding_context
from ..reliability.circuit_breaker import CircuitBreaker
from ..reliability.resilient import is_transient_anthropic, resilient_generate
from ..security.authz import AuthContext
from ..security.guardrails import SECURITY_PREAMBLE, scan_input
from .sessions import InMemorySessionStore, SessionStore


class SupportEngine:
    """Built once at startup; shared by every request. Carries no per-conversation state."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        approval_callback: ApprovalCallback | None = None,
        memory_store: MemoryStore | None = None,
        session_store: SessionStore | None = None,
        logger: StructuredLogger | None = None,
        metrics: MetricsAggregator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # Observability sinks are optional and injected — the engine works without them,
        # and tests/prod can plug in real ones.
        self.logger = logger
        self.metrics = metrics

        # --- expensive, shared, READ-ONLY-after-build machinery (the "engine") ---
        self.llm = LLMClient(self.settings)
        self.retriever = Retriever(settings=self.settings)
        self.retriever.index_knowledge_base()  # ONCE, at startup — not per request
        self.assembler = ContextAssembler()
        self.summarizer = Summarizer(self.llm)
        self.extractor = MemoryExtractor(self.llm)
        self.graph = build_support_graph(self.llm, approval_callback or auto_approve)
        self.router = ModelRouter(self.llm, self.settings)  # Phase 7: per-turn model choice

        # Phase 8: a circuit breaker for the model API. Shared across turns so sustained
        # model outages trip it and we fail fast to a graceful answer.
        self._model_breaker = CircuitBreaker(
            failure_threshold=self.settings.breaker_failure_threshold,
            reset_seconds=self.settings.breaker_reset_seconds,
        )

        # --- where state lives (injected so prod can swap in Redis / a real DB) ---
        self.memory_store: MemoryStore = memory_store or JsonFileMemoryStore(
            self.settings.memory_dir
        )
        self.session_store: SessionStore = session_store or InMemorySessionStore()

    def handle_turn(
        self, *, customer_id: str, session_id: str, message: str
    ) -> TurnResult:
        """Process one customer message. This IS the request lifecycle.

        The whole turn is one trace; each step is a span. The span tree (and the metrics
        rolled up from it) is what makes a production issue debuggable after the fact.
        """
        degraded = False
        with start_trace("chat_turn", customer_id=customer_id, session_id=session_id) as root:
            # 1. LOAD per-conversation state (what makes a stateless engine work).
            #    If the session store is unreachable, degrade to a fresh (stateless) turn
            #    rather than erroring the whole request — failure domain from Phase 4 §7.
            try:
                conversation = self.session_store.load(session_id) or ConversationMemory()
            except Exception:
                conversation = ConversationMemory()
                degraded = True
                self._log("session_load_failed", level="error")
            customer = self.memory_store.get(customer_id)

            # 2. RETRIEVE (Phase 1). If retrieval fails, degrade: answer from memory +
            #    model knowledge with no sources, instead of failing. (The grounding
            #    prompt then makes the agent more likely to defer to a human.)
            with span("retrieve") as sp:
                try:
                    chunks = self.retriever.retrieve(message)
                    retrieved_context = build_grounding_context(chunks)
                    sp.set(num_chunks=len(chunks), sources=[c.chunk.source for c in chunks])
                except Exception:
                    chunks = []
                    retrieved_context = ""
                    degraded = True
                    sp.set(degraded=True)
                    self._log("retrieval_failed", level="error")

            # 3. COMPACT if over budget (Phase 2).
            conversation.add_user(message)
            with span("compact") as sp:
                summarized = compact_if_needed(
                    conversation,
                    self.summarizer,
                    budget_tokens=self.settings.context_budget_tokens,
                    keep_recent=self.settings.keep_recent_messages,
                )
                sp.set(summarized=summarized)

            # 4. ASSEMBLE the prompt (Phase 2).
            with span("assemble") as sp:
                assembled = self.assembler.assemble(
                    customer=customer,
                    conversation=conversation,
                    retrieved_context=retrieved_context,
                )
                sp.set(est_tokens=assembled.estimated_tokens)

            # 4b. ROUTE (Phase 7): pick the cheapest model that can handle this turn.
            with span("route") as sp:
                model = self.router.choose(message)
                sp.set(model=model)

            # 4c. GUARDRAIL (Phase 9): screen the input for injection attempts. This is a
            #     signal (log + harden), NOT the security boundary — authz is (step 5).
            with span("guardrail") as sp:
                verdict = scan_input(message)
                sp.set(suspicious=verdict.suspicious)
                if verdict.suspicious:
                    self._log("suspicious_input", level="warning", reasons=verdict.reasons)
            # Always-on hardening preamble; the authenticated caller authorizes every tool.
            secured_system = assembled.system + TOOL_GUIDANCE + SECURITY_PREAMBLE
            auth = AuthContext(customer_id)

            # 5. GENERATE (Phase 3 graph) with Phase 8 resilience: try the routed model,
            #    fall back to a second model on transient failure, gate with a circuit
            #    breaker, and degrade to a graceful answer if all of it fails.
            with span("generate") as sp:
                models = [model]
                if self.settings.fallback_model != model:
                    models.append(self.settings.fallback_model)

                def run_on(m: str):
                    return run_agent_turn(
                        self.graph,
                        messages=list(assembled.messages),
                        system=secured_system,
                        model=m,
                        auth=auth,
                    )

                outcome = resilient_generate(
                    run_on,
                    models=models,
                    is_transient=is_transient_anthropic,
                    breaker=self._model_breaker,
                    on_event=lambda ev, f: self._log(ev, level="warning", **f),
                )
                answer, usage = outcome.answer, outcome.usage
                steps, tools_called = outcome.steps, outcome.tools_called
                degraded = degraded or outcome.degraded
                sp.set(model=outcome.model, tools_called=tools_called, degraded=outcome.degraded)
            conversation.add_assistant(answer)

            # 6. REMEMBER: extract durable facts into long-term memory. Best-effort —
            #    a failure here must not lose the answer we already produced.
            with span("remember") as sp:
                new_facts: list[str] = []
                if not degraded:
                    try:
                        for fact in self.extractor.extract(message, answer):
                            if customer.absorb_fact(fact):
                                new_facts.append(fact)
                    except Exception:
                        self._log("memory_extract_failed", level="error")
                sp.set(new_facts=len(new_facts))

            # 7. SAVE state back to the stores. Best-effort — a save failure degrades
            #    future continuity but must not fail the response we're returning.
            try:
                self.session_store.save(session_id, conversation)
                if new_facts or customer.name:
                    self.memory_store.save(customer)
            except Exception:
                degraded = True
                self._log("state_save_failed", level="error")

        # The turn (and its trace) is done — roll it up for logs + metrics.
        summary = summarize_trace(root)
        if self.logger is not None:
            self.logger.log("turn_complete", **summary)
        if self.metrics is not None:
            self.metrics.record(summary)

        return TurnResult(
            answer=answer,
            retrieved=[f"{c.chunk.source} > {c.chunk.section}" for c in chunks],
            summarized=summarized,
            new_facts=new_facts,
            usage=usage,
            steps=steps,
            tools_called=tools_called,
            context=retrieved_context,
            trace=root,
            degraded=degraded,
        )

    def _log(self, event: str, *, level: str = "info", **fields) -> None:
        if self.logger is not None:
            self.logger.log(event, level=level, **fields)
